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

def transfer_motion(frame1, frame2, frame3, flow=None):
  ''' uses dense optical flow and inverse flow to transfer motion from (frame1, frame2) to frame3'''
  # Convert the frames to grayscale
  gray1 = cv2.cvtColor(np.asarray(frame1), cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(np.asarray(frame2), cv2.COLOR_BGR2GRAY)

  # Compute the dense optical flow from frame1 to frame2
  prev_flow = flow
  flow = cv2.calcOpticalFlowFarneback(gray1, gray2, prev_flow, 0.5, 3, 15, 3, 5, 1.2, 0)

  # Compute the inverse flow from frame2 to frame1
  inv_flow = -flow

  # Warp frame3 using the inverse flow
  h, w = gray1.shape[:2]
  x, y = np.meshgrid(np.arange(w), np.arange(h))
  x_inv = np.round(x + inv_flow[...,0]).astype(np.float32)
  y_inv = np.round(y + inv_flow[...,1]).astype(np.float32)
  x_inv = np.clip(x_inv, 0, w-1)
  y_inv = np.clip(y_inv, 0, h-1)
  warped = cv2.remap(np.asarray(frame3), x_inv, y_inv, cv2.INTER_LINEAR)

  return PIL.Image.fromarray(warped), flow

"""
------------------------------------------------------------------------------------------------------------------------
"""

import torch
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline

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
@click.option('--prompt', type=str, default=None)
@click.option('--negative-prompt', type=str, default=None)
@click.option('--prompt-strength', type=float, default=7.0)
#@click.option('--scheduler', type=click.Choice(['default']), default='default')
@click.option('--num-inference-steps', type=int, default=10)
@click.option('--controlnet', type=click.Choice(['hed', 'canny', 'scribble', 'openpose']), default='hed')
@click.option('--controlnet-strength', type=float, default=1.0)
@click.option('--fix-orientation', is_flag=True, default=True)
@click.option('--consistency-strength', type=float, default=0.5, help="Weight of the consistency, or how much of the predicted output frame is fed forward and mixed with the initial noise latent used to diffuse each frame (0.0 = no consistency, 1.0 = full consistency)")

def main(input_video, output_video, prompt, negative_prompt, num_inference_steps, prompt_strength, controlnet, controlnet_strength, fix_orientation, consistency_strength):
  # run controlnet pipeline on video

  from controlnet_aux import CannyDetector, OpenposeDetector, MLSDdetector, HEDdetector, MidasDetector

  detector_kwargs = dict({
  #   "low_threshold": 100,
  #    "high_threshold": 200,
  })
  #detector_model = CannyDetector() #HEDdetector.from_pretrained("lllyasviel/ControlNet")
  detector_model = HEDdetector.from_pretrained("lllyasviel/ControlNet")
  controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)

  pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
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

    generator = None #torch.manual_seed(0)

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
      # Calculates dense optical flow of input by Farneback method
      #flow = cv2.calcOpticalFlowFarneback(prev_input_gray, input_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
      # apply the flow to the previous output frame to get the predicted next output frame
      #prev_output = np.asarray(prev_output_frame)
      #print(f'input_gray.shape="{input_gray.shape}", min={np.min(input_gray)}, max={np.max(input_gray)}')
      #print(f'prev_output_frame.shape="{prev_output_frame.shape}", min={np.min(prev_output_frame)}, max={np.max(prev_output_frame)}')
      predicted_output_frame, flow = transfer_motion(prev_input_frame, input_frame, prev_output_frame, flow)
      predicted_output_frame.save("predicted_output_frame.png")
      print(f"predicted_output_frame.shape={predicted_output_frame.size}")
      print(f"flow.shape={flow.shape}, minx={np.min(flow[:,:,0])}, maxx={np.max(flow[:,:,0])}, miny={np.min(flow[:,:,1])}, maxy={np.max(flow[:,:,1])}")
      """
      defheight, defwidth = pipe._default_height_width(None, None, input_frame)
      print(f'default WxH: {defwidth}x{defheight}')
      # convert the predicted output frame to latent
      predicted_output_frame = preprocess(predicted_output_frame)
      noisy_latent, orig_latent, noise = prepare_latents(pipe, predicted_output_frame, get_timesteps(pipe, num_inference_steps, consistency_weight, pipe._execution_device)[0], generator=generator)
      predicted_output_frame = predicted_output_frame.to(device=pipe.vae.device, dtype=torch.float16)
      predicted_output_frame = torch.nn.functional.interpolate(predicted_output_frame, size=(height // pipe.vae_scale_factor // 256 * 128, width // pipe.vae_scale_factor // 256 * 128) )
      predicted_output_latent = pipe.vae.encode(predicted_output_frame)
      # blend the latent with the noise using linear interpolation according to consistency_weight
      noisy_latent = (1 - consistency_weight) * noisy_latent + consistency_weight * predicted_output_latent
      """
      '''
      # 5. set timesteps
      batch_size = 1
      num_images_per_prompt = 1
      pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)
      timesteps, num_inference_steps = pipe.get_timesteps(num_inference_steps, consistency_strength, pipe.device)
      latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
      
      # 6. Prepare latent variables
      num_channels_latents = pipe.unet.in_channels
      latents = prepare_latents(
          preprocess(PIL.Image.fromarray(np.uint8(predicted_output_frame)).convert("RGB")),
          latent_timestep,
          batch_size * num_images_per_prompt,
          num_channels_latents,
          height,
          width,
          pipe.dtype,
          pipe.device,
          generator,
          latents,
      )
      '''
      
    else:
      predicted_output_frame = None #PIL.Image.new("RGB", (width, height), 'black')
    
    if predicted_output_frame != None:
      print(f'predicted_output_frame WxH: "{predicted_output_frame.width}x{predicted_output_frame.height}"')
    print(f'input_frame WxH: "{input_image.width}x{input_image.height}"')

    input_image.save("input_image.png")
    control_image = detector_model(input_image, **detector_kwargs).convert("RGB")
    control_image.save("control_image.png")
    #predicted_output_frame.save("predicted_output_frame.png")

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
      image = predicted_output_frame if prev_output_frame != None else PIL.Image.new("RGB", (width, height), 'gray'),
      strength = (1.0 - consistency_strength) if prev_output_frame != None else 1.0
    ).images[0]

    output_frame.save("frame.png")
    prev_input_frame = input_image.copy()
    prev_input_gray = input_gray
    prev_output_frame = output_frame.copy()
    prev_predicted_output_frame = predicted_output_frame
    return output_frame
  
  process_frames(input_video, output_video, frame_filter)

if __name__ == "__main__":
  main()
