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
from typing import Tuple, List, FrozenSet, Sequence, MutableSequence, Mapping, Optional, Any, Type, Union
from controlnet_aux import CannyDetector, OpenposeDetector, MLSDdetector, HEDdetector, MidasDetector

def transfer_motion(srcframe1, srcframe2, destframe, flow:Optional[np.ndarray] = None) -> \
    Tuple[PIL.Image.Image, np.ndarray]:
  '''
  predict next output frame dense optical flow estimate to transfer motion between srcframe1 and srcframe2 
  and optionally prior motion vector flow onto previous destframe at time of srcframe1
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

class MidasDetectorWrapper:
  ''' a wrapper around the midas detector model which allows 
  choosing either the depth or the normal output on creation '''
  def __init__(self, model=None, model_type="dpt_hybrid", output_index=0, **kwargs):
    self.model = model if model != None else MidasDetector(model_type=model_type)
    self.output_index = output_index
    self.default_kwargs = dict(kwargs)
  def __call__(self, image, **kwargs):
    ka = dict(list(self.default_kwargs.items()) + list(kwargs.items()))
    return self.model(image, **ka)[self.output_index]


class NormalDetector:
  def __init__(self, bg_threshold=0.4):
    raise NotImplementedError("NormalDetector is not working yet")
    from transformers import pipeline
    self.depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas" )
    self.bg_threshold = bg_threshold
  
  def __call__(self, image, bg_threshold=None):
    if bg_threshold is None:
      bg_threshold = self.bg_threshold
    
    image = self.depth_estimator(image)['predicted_depth'][0]
    image = image.numpy()

    image_depth = image.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)

    bg_threhold = self.bg_threshold

    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0

    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0

    z = np.ones_like(x) * np.pi * 2.0

    image = np.stack([x, y, z], axis=2)
    image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image


def stackh(images):
  cellh = max([image.height for image in images])
  cellw = max([image.width for image in images])
  result = PIL.Image.new("RGB", (cellw * len(images), cellh), "black")
  for i, image in enumerate(images):
    result.paste(image.convert("RGB"), (i * cellw + (cellw - image.width)//2, (cellh - image.height)//2))
  #print(f"stackh: {len(images)}, cell WxH: {cellw}x{cellh}, result WxH: {result.width}x{result.height}")
  return result

def expanddims(*sides):
  from typing import Tuple, Iterable, String
  if not (isinstance(sides, Iterable) and not isinstance(sides, String)):
    sides = [sides]
  if len(sides) == 1:
    sides = [sides[0], sides[0], sides[0], sides[0]]
  if len(sides) == 2:
    sides = [sides[0], sides[1], sides[0], sides[1]]
  if len(sides) == 3:
    sides = [sides[0], sides[1], sides[2], sides[1]]
  return sides

def roundrect(size, radius:Tuple[int,int,int,int], border:Tuple[int,int,int,int], fill="white", outline="black"):
  from PIL import Image, ImageDraw
  width, height = size
  tl, tr, br, bl = radius
  tl = min(tl, width//2, height//2)
  tr = min(tr, width//2, height//2)
  bl = min(bl, width//2, height//2)
  br = min(br, width//2, height//2)
  btl, btr, bbr, bbl = border
  btl = min(btl, width//2, height//2)
  btr = min(btr, width//2, height//2)
  bbl = min(bbl, width//2, height//2)
  bbr = min(bbr, width//2, height//2)
  result = PIL.Image.new("RGBA", size, color=fill)
  draw = ImageDraw.Draw(result)
  draw.rectangle((0,0,width,height), fill=fill, outline=outline)
  draw.rectangle((btl, btl, width-btr, height-bbr), fill=None, outline=outline)
  draw.rectangle((bbl, bbl, width-bbr, height-bbr), fill=None, outline=outline)
  draw.pieslice((0, 0, tl*2, tl*2), 180, 270, fill=None, outline=outline)
  draw.pieslice((width-tr*2, 0, width, tr*2), 270, 360, fill=None, outline=outline)
  draw.pieslice((0, height-bl*2, bl*2, height), 90, 180, fill=None, outline=outline)
  draw.pieslice((width-br*2, height-br*2, width, height), 0, 90, fill=None, outline=outline)
  return result

def textbox(s, font, color, padding=(1,1,1,1), border=(0,0,0,0), corner_radius=(2,2,2,2), background_color="white", border_color="black"):
  text = PIL.Image.new('RGBA', fontgetsize(str), background_color)
  draw = PIL.ImageDraw.Draw(text)
  draw.text((0, 0), str, font=font, fill=color)
  return text

def rgbtohsl(rgb:np.ndarray):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  r, g, b = r / 255.0, g / 255.0, b / 255.0
  mx = np.amax([r, g, b])
  mn = np.amin([r, g, b])
  df = mx-mn
  if mx == mn: h = 0
  elif mx == r: h = (60 * ((g-b)/df) + 360) % 360
  elif mx == g: h = (60 * ((b-r)/df) + 120) % 360
  elif mx == b: h = (60 * ((r-g)/df) + 240) % 360
  if mx == 0: s = 0
  else: s = df/mx
  l = (mx+mn)/2
  return np.array([h, s, l])

def brightcontrastmatch(source, template):
  oldshape = source.shape
  source = source.ravel()
  template = template.ravel()
  s_mean, s_std = source.mean(), source.std()
  t_mean, t_std = template.mean(), template.std()
  source = (source - s_mean) * (t_std / s_std) + t_mean
  return source.reshape(oldshape)

def avghuesatmatch(source, template):
  oldshape = source.shape
  source = source.ravel()
  template = template.ravel()
  s_hsl = np.asarray(PIL.Image.fromarray(source, mode="RGB").convert(mode="HSL"))
  t_hsl = np.asarray(PIL.Image.fromarray(template, mode="RGB").convert(mode="HSL"))
  s_hue, s_sat = s_hsl[:,0], s_hsl[:,1]
  t_hue, t_sat = t_hsl[:,0], t_hsl[:,1]
  s_hue_mean, s_hue_std = s_hue.mean(), s_hue.std()
  s_sat_mean, s_sat_std = s_sat.mean(), s_sat.std()
  t_hue_mean, t_hue_std = t_hue.mean(), t_hue.std()
  t_sat_mean, t_sat_std = t_sat.mean(), t_sat.std()
  s_hue = (s_hue - s_hue_mean) * (t_hue_std / s_hue_std) + t_hue_mean
  s_sat = (s_sat - s_sat_mean) * (t_sat_std / s_sat_std) + t_sat_mean
  s_hsl[:,0], s_hsl[:,1] = s_hue, s_sat
  return (PIL.Image.fromarray(s_hsl, mode="HSL")).reshape(oldshape)

def histomatch(source, template):
  oldshape = source.shape
  source = source.ravel()
  template = template.ravel()
  s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
  t_values, t_counts = np.unique(template, return_counts=True)
  s_quantiles = np.cumsum(s_counts).astype(np.float64)
  s_quantiles /= s_quantiles[-1]
  t_quantiles = np.cumsum(t_counts).astype(np.float64)
  t_quantiles /= t_quantiles[-1]
  interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
  return interp_t_values[bin_idx].reshape(oldshape)

# ----------------------------------------------------------------------------------------------

def process_frames(input_video, output_video, wrapped, start_time=None, end_time=None, duration=None, max_dimension=None, min_dimension=None, round_dims_to=None, fix_orientation=False):
    from moviepy import editor as mp
    from PIL import Image
    from tqdm import tqdm
    # Load the video
    video = mp.VideoFileClip(input_video)
    # scale the video frames if a min/max size is given
    if fix_orientation:
      h, w = video.size
    else:
      w, h = video.size
    
    if max_dimension != None:
      if w > h:
        w, h = max_dimension, int(h / w * max_dimension)
      else:
        w, h = int(w / h * max_dimension), max_dimension
    if min_dimension != None:
      if w < h:
        w, h = min_dimension, int(h / w * min_dimension)
      else:
        w, h = int(w / h * min_dimension), min_dimension
    if round_dims_to is not None:
      w = round_dims_to * (w // round_dims_to)
      h = round_dims_to * (h // round_dims_to)
    # set the start and end time and duration to process if given
    if end_time is not None:
      video = video.subclip(0, end_time)
    if start_time is not None:
      video = video.subclip(start_time)
    if duration != None:
      video = video.subclip(0, duration)
    # Create a new video with the processed frames
    processed_video = None
    try:
      processed_video = mp.ImageSequenceClip([
        np.array(wrapped(framenum, Image.fromarray(frame).resize((w,h)))) 
          for framenum, frame in 
            enumerate(tqdm(video.iter_frames()))
        ], fps=video.fps)
    finally:
      # save the video
      if processed_video != None:
        processed_video.write_videofile(output_video)

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.option('--start-time', type=Optional[Union[float, str]], default=None, help="start time in seconds")
@click.option('--end-time', type=Optional[Union[float, str]], default=None, help="end time in seconds")
@click.option('--duration', type=Optional[Union[float, str]], default=None, help="duration in seconds")
@click.option('--max-dimension', type=int, default=None, help="maximum dimension of the video")
@click.option('--min-dimension', type=int, default=None, help="minimum dimension of the video")
@click.option('--round-dims-to', type=int, default=None, help="round the dimensions to the nearest multiple of this number")
@click.option('--prompt', type=str, default=None, help="prompt used to guide the denoising process")
@click.option('--negative-prompt', type=str, default=None, help="negative prompt, can be used to prevent the model from generating certain words")
@click.option('--prompt-strength', type=float, default=7.0, help="how much influence the prompt has on the output")
#@click.option('--scheduler', type=click.Choice(['default']), default='default', help="which scheduler to use")
@click.option('--num-inference-steps', '--steps', type=int, default=10, help="number of inference steps, depends on the scheduler, trades off speed for quality. 20-50 is a good range from fastest to best.")
@click.option('--controlnet', type=click.Choice(['hed', 'canny', 'scribble', 'openpose', 'depth', 'normal', 'mlsd']), default='hed', help="which pretrained controlnet annotator to use")
@click.option('--controlnet-strength', type=float, default=1.0, help="how much influence the controlnet annotator's output is used to guide the denoising process")
@click.option('--fix-orientation', is_flag=True, default=True, help="resize videos shot in portrait mode on some devices to fix incorrect aspect ratio bug")
@click.option('--init-image-strength', type=float, default=0.5, help="the init-image strength, or how much of the prompt-guided denoising process to skip in favor of starting with an existing image")
@click.option('--feedthrough-strength', type=float, default=0.0, help="the init-image is composed by transferring motion from the input video onto each output frame, and blending it optionally with the frame input that is sent to the controlnet detector")
@click.option('--show-detector', is_flag=True, default=False, help="show the controlnet detector output")
@click.option('--show-input', is_flag=True, default=False, help="show the input frame")
@click.option('--show-output', is_flag=True, default=True, help="show the output frame")
@click.option('--show-flow', is_flag=True, default=False, help="show the motion transfer (not implemented yet)")
@click.option('--dump-frames', type=click.Path(), default=None, help="write frames to a file during processing to visualise progress. may contain {} placeholder for frame number")
def main(input_video, output_video, start_time, end_time, duration, max_dimension, min_dimension, round_dims_to, prompt, negative_prompt, prompt_strength, num_inference_steps, controlnet, controlnet_strength, fix_orientation, init_image_strength, feedthrough_strength, show_detector, show_input, show_output, show_flow, dump_frames):   
  # run controlnet pipeline on video

  if controlnet == 'canny':
    detector_kwargs = dict({
      "low_threshold": int(255//(3/1)),
      "high_threshold": int(255//(3/2))
    })
    detector_model = CannyDetector()
    controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
  elif controlnet == 'openpose':
    detector_kwargs = dict()
    detector_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    controlnet_model = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16)
  elif controlnet == 'depth':
    detector_kwargs = dict({'bg_th': 0.1})
    detector_model = MidasDetectorWrapper(model_type="dpt_hybrid") #MidasDetectorWrapper(None, 0)
    controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
  elif controlnet == 'mlsd':
    detector_kwargs = dict()
    detector_model = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
    controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16)
  elif controlnet == 'hed':
    detector_kwargs = dict()
    detector_model = HEDdetector.from_pretrained("lllyasviel/ControlNet")
    controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
  elif controlnet == 'normal':
    detector_kwargs = dict()
    detector_model = MidasDetectorWrapper(None, 1)
    #detector_model = NormalDetector()
    controlnet_model = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16)
  else:
    raise NotImplementedError("controlnet type not implemented")
  
  pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet_model,
    torch_dtype=torch.float16
  )

  #if scheduler.lower().startswith("unipcm"):
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
    # state kept between frames
    nonlocal prev_input_frame
    nonlocal prev_input_gray
    nonlocal prev_output_frame
    nonlocal prev_predicted_output_frame
    nonlocal flow

    #print(f"Processing frame {framenum} {input_frame.width}x{input_frame.height} shape={np.asarray(input_frame).shape}...")
    print()
    
    generator = None

    #if fix_orientation:
    #  input_frame = input_frame.resize((input_frame.height//256*128, input_frame.width//256*128), PIL.Image.BICUBIC)
    #else:
    #  input_frame = input_frame.resize((input_frame.width//256*128, input_frame.height//256*128), PIL.Image.BICUBIC)

    width = input_frame.width
    height = input_frame.height
    input_image = input_frame.copy()
    input_frame = np.asarray(input_image)

    # Converts each frame to grayscale as required by optical flow
    input_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
    
    # if we have a previous frame, transfer motion from the input to the output and blend with the noise
    if prev_input_frame != None:
      predicted_output_frame, flow = transfer_motion(prev_input_frame, input_frame, prev_output_frame, flow)
    else:
      predicted_output_frame = None 
    
    control_image = detector_model(input_image, **detector_kwargs)
    ci = np.asarray(control_image)
    print(f"ci.shape={ci.shape} ci.min={ci.min()} ci.max={ci.max()} ci.mean={ci.mean()} ci.std={ci.std()}")

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
      controlnet_conditioning_image=control_image,
      controlnet_conditioning_scale=controlnet_strength,
      image = init_image,
      strength = 1.0 - strength,
    ).images[0]

    final_frame = stackh( list( 
                         list([input_image] if show_input else []) + 
                         list([control_image] if show_detector else []) +
                         list([output_frame] if show_output else [])) )

    if dump_frames != None:
      final_frame.save(dump_frames.format(framenum)),
    #output_frame.save("frame.png")
    prev_input_frame = input_image.copy()
    prev_input_gray = input_gray
    prev_output_frame = output_frame.copy()
    prev_predicted_output_frame = predicted_output_frame
    return final_frame
  
  process_frames(input_video, output_video, frame_filter, start_time, end_time, duration, max_dimension, min_dimension, round_dims_to, fix_orientation)

if __name__ == "__main__":
  main()


