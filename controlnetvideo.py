"""
!pip install -q transformers xformers accelerate git+https://github.com/huggingface/accelerate.git
!pip install -q git+https://github.com/mikegarts/diffusers.git@stablediffusion.controlnet.img2img.pipeline

"""

import wrapt
import click
import numpy as np
import torch
import PIL.Image
import cv2
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from typing import Tuple, List, FrozenSet, Sequence, MutableSequence, Mapping, Optional, Any, Type

def transfer_motion(srcframe1, srcframe2, destframe, flow:Optional[np.ndarray] = None) -> \
    Tuple[PIL.Image.Image, np.ndarray]:
  '''
  uses inverse dense optical flow to transfer motion between frame1 and frame2 (and using prior motion vector f) onto frame3
  '''
  
  # Convert the frames to grayscale
  gray1 = cv2.cvtColor(np.asarray(srcframe1), cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(np.asarray(srcframe2), cv2.COLOR_BGR2GRAY)

  # Compute the dense optical flow from frame1 to frame2
  prev_flow = flow
  flow = cv2.calcOpticalFlowFarneback(gray1, gray2, prev_flow, 0.5, 3, 15, 3, 7, 1.5, 0)

  # Compute the inverse flow from frame2 to frame1
  inv_flow = -flow

  # Warp frame3 using the inverse flow
  h, w = gray1.shape[:2]
  x, y = np.meshgrid(np.arange(w), np.arange(h))
  x_inv = np.round(x + inv_flow[...,0]).astype(np.float32)
  y_inv = np.round(y + inv_flow[...,1]).astype(np.float32)
  x_inv = np.clip(x_inv, 0, w-1)
  y_inv = np.clip(y_inv, 0, h-1)
  warped = cv2.remap(np.asarray(destframe), x_inv, y_inv, cv2.INTER_LINEAR)

  return PIL.Image.fromarray(warped), flow

def stackh(images):
  cellh = max([image.height for image in images])
  cellw = max([image.width for image in images])
  result = PIL.Image.new("RGB", (cellw * len(images), cellh), "black")
  for i, image in enumerate(images):
    result.paste(image.convert("RGB"), (i * cellw + (cellw - image.width)//2, (cellh - image.height)//2))
  print(f"stackh: {len(images)}, cell WxH: {cellw}x{cellh}, result WxH: {result.width}x{result.height}")
  return result

def process_frames(input_video, output_video, wrapped):
    from moviepy import editor as mp
    from PIL import Image

    # Load the video
    video = mp.VideoFileClip(input_video)
    # Create a new video with the processed frames
    processed_video = mp.ImageSequenceClip([np.array(wrapped(framenum, Image.fromarray(frame))) for framenum, frame in enumerate(video.iter_frames())], fps=video.fps)
    # save the video
    processed_video.write_videofile(output_video)


@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.option('--prompt', type=str, default=None, help="prompt used to guide the denoising process")
@click.option('--negative-prompt', type=str, default=None, help="negative prompt, can be used to prevent the model from generating certain words")
@click.option('--prompt-strength', type=float, default=7.0, help="how much influence the prompt has on the output")
#@click.option('--scheduler', type=click.Choice(['default']), default='default', help="which scheduler to use")
@click.option('--num-inference-steps', '--steps', type=int, default=10, help="number of inference steps, depends on the scheduler, trades off speed for quality. 20-50 is a good range from fastest to best.")
@click.option('--controlnet', type=click.Choice(['hed', 'canny', 'scribble', 'openpose', 'midas', 'normal']), default='hed', help="which pretrained controlnet annotator to use")
@click.option('--controlnet-strength', type=float, default=1.0, help="how much influence the controlnet annotator's output is used to guide the denoising process")
@click.option('--fix-orientation', is_flag=True, default=True, help="resize videos shot in portrait mode on some devices to fix incorrect aspect ratio bug")
@click.option('--init-image-strength', type=float, default=0.5, help="the init-image strength, or how much of the prompt-guided denoising process to skip in favor of starting with an existing image")
@click.option('--feedthrough-strength', type=float, default=0.0, help="the init-image is composed by transferring motion from the input video onto each output frame, and blending it optionally with the frame input that is sent to the controlnet detector")
@click.option('--show-detector', is_flag=True, default=False, help="show the controlnet detector output")
@click.option('--show-input', is_flag=True, default=False, help="show the input frame")
@click.option('--show-output', is_flag=True, default=False, help="show the output frame")
@click.option('--show-flow', is_flag=True, default=False, help="show the motion transfer (not implemented yet)")
@click.option('--save-intermediate-frames', type=click.Path(), default=None, help="write frames to a file during processing to visualise progress")
def main(input_video, output_video, prompt, negative_prompt, prompt_strength, num_inference_steps, controlnet, controlnet_strength, fix_orientation, init_image_strength, feedthrough_strength, show_detector, show_input, show_output, show_flow, save_intermediate_frames):

  # run controlnet pipeline on video

   from controlnet_aux import CannyDetector, OpenposeDetector, MLSDdetector, HEDdetector, MidasDetector

  if controlnet == 'canny':
    detector_kwargs = dict({
      "low_threshold": 100,
      "high_threshold": 200,
    })
    detector_model = CannyDetector()
    controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
  elif controlnet == 'openpose':
    detector_kwargs = dict()
    detector_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
  elif controlnet == 'midas':
    detector_kwargs = dict()
    detector_model = MidasDetector.from_pretrained("lllyasviel/ControlNet")
    controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-midas", torch_dtype=torch.float16)
  elif controlnet == 'normal':
    detector_kwargs = dict()
    detector_model = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
    controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16)
  elif controlnet == 'hed':
    detector_kwargs = dict()
    detector_model = HEDdetector.from_pretrained("lllyasviel/ControlNet")
    controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)

  pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet_model,
    torch_dtype=torch.float16
  )

  pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
  pipe.enable_xformers_memory_efficient_attention()
  pipe.enable_model_cpu_offload()
  pipe.run_safety_checker = lambda image, text_embeds, text_embeds_negative: (image, False)

  prev_input_frame = None
  prev_output_frame = None
  prev_input_gray = None
  prev_predicted_output_frame = None
  flow = None

  def frame_filter(framenum, input_frame):
    nonlocal prev_input_frame
    nonlocal prev_input_gray
    nonlocal prev_output_frame
    nonlocal prev_predicted_output_frame
    nonlocal flow

    print(f"Processing frame {framenum} {input_frame.width}x{input_frame.height} shape={np.asarray(input_frame).shape}...")

    generator = None

    if fix_orientation:
      input_frame = input_frame.resize((input_frame.height//256*128, input_frame.width//256*128), PIL.Image.BICUBIC)
    width = input_frame.width
    height = input_frame.height
    input_image = input_frame.copy()
    input_frame = np.asarray(input_image) #.astype(np.float32)/127.5 - 1.0

    # Converts each frame to grayscale - we previously 
    # only converted the first frame to grayscale
    input_gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    
    # generate noisy latent variables
    #shape = (1, pipe.unet.in_channels, width // pipe.vae_scale_factor, height // pipe.vae_scale_factor)    #noisy_latent = torch.randn(shape, device=pipe.device, dtype=torch.float16) * pipe.scheduler.init_noise_sigma    #noisy_latent = pipe.prepare_latents(1, pipe.unet.in_channels, height, width, torch.float16, pipe.device, generator=generator)    #shape = noisy_latent.shape    #print(f"noisy latent shape={shape}")

    # if we have a previous frame, transfer motion from the input to the output and blend with the noise
    if prev_input_frame != None:
      predicted_output_frame, flow = transfer_motion(prev_input_frame, input_frame, prev_output_frame, flow)
    else:
      predicted_output_frame = None #PIL.Image.new("RGB", (width, height), 'black')
    
    #if predicted_output_frame != None:
    #  print(f'predicted_output_frame WxH: "{predicted_output_frame.width}x{predicted_output_frame.height}"')
    #print(f'input_frame WxH: "{input_image.width}x{input_image.height}"')

    #input_image.save("input_image.png")
    control_image = detector_model(input_image, **detector_kwargs).convert("RGB")
    #control_image.save("control_image.png")
    #predicted_output_frame.save("predicted_output_frame.png")

    if predicted_output_frame == None:
      strength = feedthrough_strength * init_image_strength
      init_image = input_image.copy()
    else:
      strength = init_image_strength
      init_image = PIL.Image.blend(input_image, predicted_output_frame, 1.0 - feedthrough_strength)

    # run the pipeline
    output_frame = pipe(
      prompt=prompt, 
      guidance_scale=prompt_strength,
      negative_prompt=negative_prompt, 
      num_inference_steps=num_inference_steps, 
      generator=generator,
      #latents=noisy_latent,
      controlnet_conditioning_image=control_image,
      controlnet_conditioning_scale=controlnet_strength,
      image = init_image,
      strength = 1.0 - strength
    ).images[0]

    final_frame = stackh( list( 
                         list([input_image] if show_input else []) + 
                         list([control_image] if show_detector else []) +
                         list([output_frame] if show_output else [])) )

    if save_intermediate_frames != None:
      final_frame.save(save_intermediate_frames.format(framenum)),
    #output_frame.save("frame.png")
    prev_input_frame = input_image.copy()
    prev_input_gray = input_gray
    prev_output_frame = output_frame.copy()
    prev_predicted_output_frame = predicted_output_frame
    return final_frame
  
  process_frames(input_video, output_video, frame_filter)

if __name__ == "__main__":
  main()

