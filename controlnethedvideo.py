import wrapt
import click
import numpy as np
import torch
import PIL.Image
import cv2

def process_frames(input_video, output_video, wrapped):
    from moviepy import editor as mp
    from PIL import Image

    # Load the video
    video = mp.VideoFileClip(input_video)
    # Create a new video with the processed frames
    processed_video = mp.ImageSequenceClip([np.array(wrapped(framenum, Image.fromarray(frame))) for framenum, frame in enumerate(video.iter_frames())], fps=video.fps)
    # save the video
    processed_video.write_videofile(output_video)

def preprocess(image):
    ''' copied from pipeline_stable_diffusion_img2img.py '''
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h)))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
def get_timesteps(pipe, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = pipe.scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start

def prepare_latents(pipe, image, timestep, batch_size=1, num_images_per_prompt=1, dtype=None, device=None, generator=None):
    if device is None:
        device = pipe._execution_device
    if dtype is None:
        dtype = torch.float16
    image = image.to(device=pipe.vae.device, dtype=dtype)
    init_latent_dist = pipe.vae.encode(image).latent_dist
    init_latents = init_latent_dist.sample(generator=generator)
    init_latents = pipe.vae.config.scaling_factor * init_latents

    # Expand init_latents for batch_size and num_images_per_prompt
    init_latents = torch.cat([init_latents] * batch_size * num_images_per_prompt, dim=0)
    init_latents_orig = init_latents

    # add noise to latents using the timesteps
    noise = torch.tensor(torch.randn(init_latents.shape, generator=generator, dtype=dtype)).cpu()
    init_latents = pipe.scheduler.add_noise(init_latents.cpu(), noise, timestep)
    latents = init_latents
    return latents, init_latents_orig, noise


@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.option('--prompt', type=str, default=None)
@click.option('--negative_prompt', type=str, default=None)
@click.option('--prompt_guidance_scale', type=float, default=7.0)
@click.option('--scheduler', type=click.Choice(['default']), default='default')
@click.option('--num_inference_steps', type=int, default=10)
@click.option('--controlnet', type=click.Choice(['hed', 'canny', 'scribble', 'openpose']), default='hed')
@click.option('--controlnet_guidance_scale', type=float, default=1.0)
@click.option('--fix-orientation', is_flag=True, default=True)
@click.option('--consistency-weight', type=float, default=0.5, help="Weight of the consistency, or how much of the predicted output frame is fed forward and mixed with the initial noise latent used to diffuse each frame (0.0 = no consistency, 1.0 = full consistency)")
def main(input_video, output_video, prompt, negative_prompt, scheduler, num_inference_steps, prompt_guidance_scale, controlnet, controlnet_guidance_scale, fix_orientation, consistency_weight):
  # run controlnet pipeline on video

  from controlnet_aux import CannyDetector, OpenposeDetector, MLSDdetector, HEDdetector #, MidasDetector
  from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

  detector_model = HEDdetector.from_pretrained("lllyasviel/ControlNet")
  controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
  pipe = StableDiffusionControlNetPipeline.from_pretrained(
      "runwayml/stable-diffusion-v1-5", controlnet=controlnet_model, torch_dtype=torch.float16 
  )
  pipe.run_safety_checker = lambda image, text_embeds, text_embeds_negative: (image, False)

  pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

  # this command loads the individual model components on GPU on-demand.
  pipe.enable_model_cpu_offload()
  pipe.enable_xformers_memory_efficient_attention()

  prev_input_frame = None
  prev_output_frame = None
  prev_input_gray = None

  def frame_filter(framenum, input_frame):
    nonlocal prev_input_frame
    nonlocal prev_input_gray
    nonlocal prev_output_frame
    
    print(f"Processing frame {framenum} {input_frame.width}x{input_frame.height} shape={np.asarray(input_frame).shape}...")

    generator = torch.manual_seed(0)

    if fix_orientation:
      input_frame = input_frame.resize((input_frame.height, input_frame.width), PIL.Image.BICUBIC)
    width = input_frame.width
    height = input_frame.height
    input_frame = np.asarray(input_frame)

    # Converts each frame to grayscale - we previously 
    # only converted the first frame to grayscale
    input_gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    
    # generate noisy latent variables
    
    #shape = (1, pipe.unet.in_channels, width // pipe.vae_scale_factor, height // pipe.vae_scale_factor)
    #noisy_latent = torch.randn(shape, device=pipe.device, dtype=torch.float16) * pipe.scheduler.init_noise_sigma
    noisy_latent = pipe.prepare_latents(1, pipe.unet.in_channels, height, width, torch.float16, pipe.device, generator=generator)
    shape = noisy_latent.shape
    print(f"noisy latent shape={shape}")

    # if we have a previous frame, transfer motion from the input to the output and blend with the noise
    if prev_input_gray is not None:
      # Calculates dense optical flow of input by Farneback method
      flow = cv2.calcOpticalFlowFarneback(prev_input_gray, input_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
      # apply the flow to the previous output frame to get the predicted next output frame
      print(f'input_gray.shape="{input_gray.shape}"')
      predicted_output_frame = cv2.remap(np.asarray(prev_output_frame), flow, None, cv2.INTER_LINEAR)
      print(f'predicted_output_frame.shape="{predicted_output_frame.shape}"')
      # convert the predicted output frame to latent
      predicted_output_frame = preprocess(PIL.Image.fromarray(predicted_output_frame))
      noisy_latent = prepare_latents(pipe, predicted_output_frame, get_timesteps(pipe, num_inference_steps, consistency_weight, pipe._execution_device)[0], generator=generator)
      #predicted_output_frame = predicted_output_frame.to(device=pipe.vae.device, dtype=torch.float16)
      #predicted_output_frame = torch.nn.functional.interpolate(predicted_output_frame, size=(height // pipe.vae_scale_factor // 256 * 128, width // pipe.vae_scale_factor // 256 * 128) )
      #predicted_output_latent = pipe.vae.encode(predicted_output_frame)
      # blend the latent with the noise using linear interpolation according to consistency_weight
      #noisy_latent = (1 - consistency_weight) * noisy_latent + consistency_weight * predicted_output_latent


    # run the pipeline
    output_frame = pipe(
      prompt=prompt, 
      guidance_scale=prompt_guidance_scale,
      negative_prompt=negative_prompt, 
      num_inference_steps=num_inference_steps, 
      generator=generator,
      #latents=noisy_latent,
      image=detector_model(input_frame),
      #controlnet_guidance_scale=controlnet_guidance_scale
    ).images[0]

    output_frame.save("frame.png")
    prev_input_frame = input_frame
    prev_input_gray = input_gray
    prev_output_frame = output_frame
    return output_frame
  
  process_frames(input_video, output_video, frame_filter)

if __name__ == "__main__":
  main()
