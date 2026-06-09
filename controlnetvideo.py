"""
---------------------------------------------------------------------------------------------
controlnetvideo.py

Stable Diffusion Video2Video with Feedback

by Victor Condino <un1tz3r0@gmail.com>
May 21 2023 - Initial Commit
Sept 2024 - Added SDXL controlnet support
Oct 2024 - Added Flux controlnet, and sdxl reference-only, which is my favorite so far

This python script is a command line tool for rerendering videos with stable diffusion
models, making use of huggingface diffusers library and various other open source projects.
It attempts to solve the problem of frame-to-frame consistency by various methods, primarily
motion transfer from dense optical flow of the input to the output, fed back either using
controlnets or reference only attention coupling. This is not the right way to do this,
clearly the models need to be extended to allow temporal information passing between frames
and trained on video datasets, which, in the time since this scripts first incarnation, has
proven very effective. I maintain this script and the techniques it uses as a curious and
aesthetically interesting aside. Enjoy
	- V.M.C (2024/10/10)

---------------------------------------------------------------------------------------------
"""

# fix collections.MutableSequence being moved to typing
import collections, typing
setattr(collections, "MutableSequence", typing.MutableSequence)
# fix np.float deprecation
import numpy
setattr(numpy, "float", float)
setattr(numpy, "int", int)


from ctypes import ArgumentError
import click
import numpy as np
import torch
import PIL.Image
import cv2
from PIL import Image
from typing import Tuple, List, FrozenSet, Sequence, MutableSequence, Mapping, Optional, Any, Type, Union
from controlnet_aux import CannyDetector  # other detectors (openpose/hed/normal) are imported lazily in the unionxl branch
import pathlib

from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from diffusers import StableDiffusionXLControlNetUnionImg2ImgPipeline, ControlNetUnionModel

# animatable video-filter chains (parsed once, applied at several points in the frame loop)
import videofilters

# ---------------------------------------------------------------------------------------------
# audio reactivity
#
# Add a soundtrack to the animation and use MadMom's music information retrieval pretrained
# deep neural networks to sync up the animation with the music.
# ---------------------------------------------------------------------------------------------

import extract_beats

# ---------------------------------------------------------------------------------------------
# Motion estimation: pluggable dense optical flow (raft / raft_small / farneback / dis) with
# shared flow smoothing. Lives in motionestimators.py; RAFT behaviour is preserved verbatim.
# ---------------------------------------------------------------------------------------------

import torch
import motionestimators

# -----------------------------------------------------------------------------------------------
# helpers for depth-based controlnets
# -----------------------------------------------------------------------------------------------

class DPTDepthDetectorWrapper:
	''' wrapper for sdxl depth controlnet and detector models, as described in
	https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl '''

	def __init__(self):
		self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
		self.feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

	def load_controlnet(self):
		return ControlNetModel.from_pretrained(
				"diffusers/controlnet-depth-sdxl-1.0-small",
				variant="fp16",
				use_safetensors=True,
				torch_dtype=torch.float16,
		)

	def __call__(self, image):
		image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
		with torch.no_grad(), torch.autocast("cuda"):
				depth_map = self.depth_estimator(image).predicted_depth

		if False:
			depth_map = torch.nn.functional.interpolate(
				depth_map.unsqueeze(1),
				size=(1024, 1024),
				mode="bicubic",
				align_corners=False,
			)
		else:
			depth_map = depth_map.unsqueeze(1)
		depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
		depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
		depth_map = (depth_map - depth_min) / (depth_max - depth_min)
		image = torch.cat([depth_map] * 3, dim=1)
		image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
		image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
		return image

# -----------------------------------------------------------------------------------------------
# general image helpers (could be moved to a separate file)
# -----------------------------------------------------------------------------------------------

def padto(image, width, height, gravityx=0.5, gravityy=0.5):
	import PIL.ImageOps, PIL.Image
	''' pad image to width and height '''
	image = PIL.Image.fromarray(image)
	if image.size[0] < width:
		image = PIL.ImageOps.expand(image, border=(int((width - image.size[0]) / 2), 0, 0, 0), fill=0)
	if image.size[1] < height:
		image = PIL.ImageOps.expand(image, border=(0, int((height - image.size[1]) / 2), 0, 0), fill=0)
	return image

def topil(image):
	# convert to PIL.Image.Image from various types
	if isinstance(image, PIL.Image.Image):
		return image
	elif isinstance(image, np.ndarray):
		return PIL.Image.fromarray(image)
	elif isinstance(image, torch.Tensor):
		while image.ndim > 3 and image.shape[0] == 1:
			image = image[0]
		if image.ndim == 3 and image.shape[0] in [3, 1]:
			image = image.permute(1, 2, 0)
		return PIL.Image.fromarray(image.numpy())
	else:
		raise ValueError(f"cannot convert {type(image)} to PIL.Image.Image")

def stackh(images):
	''' stack images horizontally, using the largest image's height for all images '''
	images = [topil(image) for image in images]
	cellh = max([image.height for image in images])
	cellw = max([image.width for image in images])
	result = PIL.Image.new("RGB", (cellw * len(images), cellh), "black")
	for i, image in enumerate(images):
		result.paste(image.convert("RGB"), (i * cellw + (cellw - image.width)//2, (cellh - image.height)//2))
	#print(f"stackh: {len(images)}, cell WxH: {cellw}x{cellh}, result WxH: {result.width}x{result.height}")
	return result

def expanddims(*sides):
	''' takes an array of 1, 2, or 4 floating point numbers, which are interpreted
	as a single value, a horizontal and vertical value, or a top, right, bottom, left value
	and returns an array of top, right, bottom and left values, with the same value repeated
	if only one value is given, or the same values repeated twice if two values are given.
	if gravity is given, it must be either one or two values, and is used to offset the '''
	from typing import Tuple, Iterable, ByteString
	if not (isinstance(sides, Iterable) and not isinstance(sides, (str, ByteString))):
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
	import PIL.Image, PIL.ImageDraw
	def fontgetsize(s):
		draw=PIL.ImageDraw.Draw(PIL.Image.new('RGBA', (1,1), background_color))
		return draw.textsize(s, font=font)
	text = PIL.Image.new('RGBA', fontgetsize(s), background_color)
	draw = PIL.ImageDraw.Draw(text)
	draw.text((0, 0), s, font=font, fill=color)
	return text

# ---------------------------------------------------------------------------------------------
# Colorspace conversion extravaganza (these belong in a colorspace.py file for sure)
# ---------------------------------------------------------------------------------------------

def rgbtohsl(rgb:np.ndarray):
	''' vectorized rgb to hsl conversion
	input is a numpy array of shape (..., 3) and dtype float32 or uint8'''
	if rgb.dtype == np.uint8:
		rgb = rgb.astype(np.float32) / 255.0
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	#r, g, b = r / 255.0, g / 255.0, b / 255.0
	mx = np.amax(rgb, 2)
	mn = np.amin(rgb, 2)
	df = mx-mn
	h = np.zeros(r.shape)
	h[g > r] = (60 * ((g[g>r]-b[g>r])/df[g>r]) + 360) % 360
	h[b > g] = (60 * ((b[b>g]-r[b>g])/df[b>g]) + 240) % 360
	h[r > b] = (60 * ((r[r>b]-g[r>b])/df[r>b]) + 120) % 360
	# h[r == g == b] = 0
	s = np.zeros(r.shape)
	s[np.nonzero(mx)] = df[np.nonzero(mx)]/mx[np.nonzero(mx)]
	l = np.zeros(r.shape)
	l = (mx+mn)/2
	hsl = np.zeros(rgb.shape)
	hsl[:,:,0] = h
	hsl[:,:,1] = s
	hsl[:,:,2] = l
	return hsl #np.ndarray([h, s, l])

def brightcontrastmatch(source, template):
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()
	s_mean, s_std = source.mean(), source.std()
	t_mean, t_std = template.mean(), template.std()
	source = (source - s_mean) * (t_std / s_std) + t_mean
	return source.reshape(oldshape)

def avghuesatmatch(source, template):
	source = np.asarray(source)
	template = np.asarray(template)
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
	return (PIL.Image.fromarray(s_hsl.reshape(oldshape), mode="HSL")).convert(mode="RGB")

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

from skimage.exposure import match_histograms
import cv2, math

def maintain_colors(color_match_sample, prev_img, mode, amount=1.0):
		''' adjust output frame to match histogram of first output frame,
		this is how deforum does it, big thanks to them '''
		if mode == 'rgb':
				return match_histograms(prev_img, color_match_sample, multichannel=True)
		elif mode == 'hsv':
				prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
				color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
				matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
				return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
		elif mode == 'lab':
				prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
				color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
				matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True )
				return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
		else:
				raise ValueError('Invalid color mode')

# ----------------------------------------------------------------------------------------------

def process_frames(
				input_video,
				input_audio,
				output_video,
				wrapped,
				start_time=None,
				end_time=None,
				duration=None,
				rescale_time=None,
				audio_params=None, # a dict of strings, keys are parameter names, and values are PiecewiseFunctions to map the beat of the audio to changing values
				max_dimension=None,
				min_dimension=None,
				round_dims_to=None,
				fix_orientation=False,
				output_bitrate="16M",
				output_codec="libx264"):
		from moviepy.audio.AudioClip import AudioArrayClip, AudioClip
		#from moviepy import editor as mp
		import moviepy as mp
		from PIL import Image
		from tqdm import tqdm

		# Load the video
		video = mp.VideoFileClip(input_video)

		# load and analyze the audio if provided
		if input_audio != None:
			if audio_params == None or len(audio_params) == 0:
				print(f"Warning! No audio animation parameters give, input audio file will just replace input video's soundtrack")
			print(f"Analysing audio file {repr(input_audio)} for tempo and beat timings...")
			audio, audiosig, beats = extract_beats.extract_beats(input_audio, fps=video.fps)
		else:
			if not (audio_params == None or len(audio_params) == 0):
				raise ArgumentError("audio animation parameters given but no input audio file specified")
			audio, audiosig, beats = (None, None, None)

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

		# trim video to length of audio if audio is shorter
		if (not audio == None) and video.duration > len(audio)/1000.0:
			print(f"Warning: input video is {video.duration - len(audio)/1000.0} seconds longer than input audio file, *truncating video* to {len(audio)/1000.0} seconds")
			video = video.subclip(0, len(audio)/1000.0)
		elif (not audio == None) and video.duration < len(audio)/1000.0:
			print(f"Warning! input video is {len(audio)/1000.0 - video.duration} seconds longer than input audio file, *truncating audio* to {video.duration} seconds")
			audio = audio[0:video.duration*1000.0]

		audio_param_curves = {}
		if audio != None and audio_params != None:
			audio_param_curves = {k: extract_beats.parse_piecewise_function(v) for k,v in audio_params.items()}

		# Create a new video with the processed frames
		from time import monotonic
		try:
			framenum = 0
			startedat = monotonic()
			def wrapper(gf, t):
				nonlocal framenum
				nonlocal startedat
				elapsed = monotonic() - startedat
				if t > 0:
					eta = (video.duration / t) * elapsed
				else:
					eta = 0
				print(f"Processing frame {framenum} at time {t}/{video.duration} seconds... {elapsed:.2f}s elapsed, {eta:.2f}s estimated time remaining")
				# compute the animation position x (fractional bar with audio, else fraction of clip)
				# and apply it to the audio-animate curves; x is also handed to the frame filter so
				# its video-filter chains animate on the same timeline.
				frametime = framenum / video.fps
				if audio != None and beats != None:
					bar, beat = extract_beats.time_to_songpos(frametime, beats)
					anim_x = math.fmod(bar, 1.0)
				else:
					anim_x = (frametime / video.duration) if video.duration else 0.0
				params = {k: v(anim_x) for k,v in audio_param_curves.items()}
				print(f"... process_frame({framenum}, <image>, {repr(params)}) at x={anim_x:.4f}")
				result = wrapped(framenum, PIL.Image.fromarray(gf(t)).resize((w,h)),
					_anim_x=anim_x, _anim_fps=video.fps, **params)
				framenum = framenum + 1
				return np.asarray(result)

			if rescale_time == None:
				#video.fx(wrapper).write_videofile(output_video)
				videoout = video.transform(wrapper, keep_duration=True)
				if audio != None:
					videoout.with_audio(extract_beats.audio_to_clip(audio))
				videoout.write_videofile(output_video, codec=output_codec, fps=video.fps, bitrate=output_bitrate)
			else:
				raise NotImplementedError("rescale_time support is not implemented yet")
				'''
				processed_video = mp.ImageSequenceClip([
				  np.array(wrapped(framenum, Image.fromarray(frame).resize((w,h))))
				    for framenum, frame in
				      enumerate(tqdm(video.iter_frames()))
				  ], fps=video.fps)
				'''
		finally:
			# save the video
			#if processed_video != None:
			#  processed_video.write_videofile(output_video)
			video.close()

# ----------------------------------------------------------------------------------------------
# Main entry point, look at all those options!
# ----------------------------------------------------------------------------------------------

def find_unused_filename(basename, fmt="-{:d}"):
	''' add numeric suffixes to the filename if it exists until we find an unused filename to write to without overwriting anything. suffix is added after stem, just before extension. formatting may be given, defaults to "-{:d}".'''
	import os, pathlib
	curnum = 0
	basepath = pathlib.Path(basename)
	curpath = basepath
	while curpath.exists():
		curnum = curnum + 1
		curpath = pathlib.Path(basepath.parent / (basepath.stem + fmt.format(curnum) + basepath.suffix))
	return str(curpath)

@click.command()
# input and output video arguments
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.option('--overwrite/--no-overwrite', is_flag=True, default=False, help="don't overwrite existing output file -- add a numeric suffix to get a unique filename. default: --no-overwrite.")
# video timeline options
@click.option('--start-time', type=float, default=None, help="start time in seconds")
@click.option('--end-time', type=float, default=None, help="end time in seconds")
@click.option('--duration', type=float, default=None, help="duration in seconds")
@click.option('--output-bitrate', type=str, default="16M", help="output bitrate for the video, e.g. '16M'")
@click.option('--output-codec', type=str, default="libx264", help="output codec for the video, e.g. 'libx264'")
# video scaling options
@click.option('--max-dimension', type=int, default=832, help="maximum dimension of the video")
@click.option('--min-dimension', type=int, default=512, help="minimum dimension of the video")
@click.option('--round-dims-to', type=int, default=128, help="round the dimensions to the nearest multiple of this number")
@click.option('--fix-orientation/--no-fix-orientation', is_flag=True, default=True, help="resize videos shot in portrait mode on some devices to fix incorrect aspect ratio bug")
# not implemented... yet (coming soon: beat-reactive video processing):
@click.option('--no-audio', is_flag=True, default=False, help="don't include audio in the output video, even if the input video has audio")
@click.option('--audio-from', type=click.Path(exists=True), default=None, help="audio file to use for the output video, replaces the audio from the input video, will be truncated to duration of input or --duration if given. tempo and bars are analyzed and can be used to drive animation with the --audio-animate parameter.")
@click.option('--audio-offset', type=float, default=None, help="offset in seconds to start the audio from, when used with --audio-from")
@click.option('--audio-animate', type=str, default=None, help="specify parameters and curves which should be animated according to the rhythm information detected in the soundtrack. format is: 'name=L x y C x y a b c d ...; name=...; ...', where name is an animatable parameter, L is a linear transition, and C is a cubic bezier curve, x is the position within a bar (four beats, starting on the downbeat) of the audio, and y is the parameter value at that point.")
# stable diffusion options
@click.option('--prompt', type=str, default=None, help="prompt used to guide the denoising process")
@click.option('--negative-prompt', type=str, default=None, help="negative prompt, can be used to prevent the model from generating certain words")
@click.option('--prompt-strength', type=float, default=7.5, help="how much influence the prompt has on the output")
#@click.option('--scheduler', type=click.Choice(['default']), default='default', help="which scheduler to use")
@click.option('--num-inference-steps', '--steps', type=int, default=25, help="number of inference steps, depends on the scheduler, trades off speed for quality. 20-50 is a good range from fastest to best.")
@click.option('--controlnet', type=click.Choice(['unionxl', 'refxl', 'fluxcanny', 'depthxl']), default='depthxl', help="which pretrained model and conditioning to use. 'unionxl' = SDXL ControlNet-Union (one model, --control-type selects the conditioning); 'depthxl' = DPT depth + depth controlnet on SDXL; 'refxl' = SDXL reference-only; 'fluxcanny' = FLUX.1 with canny controlnet")
@click.option('--control-type', type=click.Choice(['depth', 'canny', 'openpose', 'softedge', 'normal']), default='depth', help="for --controlnet unionxl: which conditioning the ControlNet-Union model is fed (selects the detector and the union control mode)")
@click.option('--union-model', type=str, default='brad-twinkl/controlnet-union-sdxl-1.0-promax', help="huggingface repo for the SDXL ControlNet-Union model used by --controlnet unionxl")
# IP-Adapter feedback (modern replacement for reference-only): inject the motion-warped previous
# output as an image prompt to keep frame-to-frame consistency. supported with SDXL pipelines.
@click.option('--ip-adapter/--no-ip-adapter', is_flag=True, default=False, help="enable IP-Adapter image-prompt feedback (SDXL): the warped previous output frame conditions the next frame")
@click.option('--ip-adapter-scale', type=float, default=0.6, help="IP-Adapter image-prompt strength (0..1+); animatable via --audio-animate key 'ipadapter'")
@click.option('--ip-adapter-source', type=click.Choice(['feedback', 'input', 'output']), default='feedback', help="which image to use as the IP-Adapter prompt: 'feedback'=motion-warped prev output, 'input'=current input frame, 'output'=prev output (un-warped)")
@click.option('--controlnet-strength', type=float, default=1.0, help="how much influence the controlnet annotator's output is used to guide the denoising process")
@click.option('--init-image-strength', type=float, default=0.5, help="the init-image strength, or how much of the prompt-guided denoising process to skip in favor of starting with an existing image")
@click.option('--feedthrough-strength', type=float, default=0.0, help="the ratio of input to motion compensated prior output to feed through to the next frame")
# motion estimation + smoothing options
@click.option('--motion-estimator', type=click.Choice(['raft', 'raft_small', 'farneback', 'dis']), default='raft', help="dense optical flow method. 'raft' (default, GPU) is the original; 'raft_small' is lighter; 'farneback' and 'dis' are CPU (no GPU needed)")
@click.option('--motion-alpha', type=float, default=0.0, help="smooth the motion vectors over time, 0.0 is no smoothing, 1.0 is maximum smoothing")
@click.option('--motion-sigma', type=float, default=0.0, help="smooth the motion estimate spatially, 0.0 is no smoothing, used as sigma for the spatial filter")
@click.option('--motion-spatial-filter', type=click.Choice(['gaussian', 'bilateral']), default='gaussian', help="spatial flow smoothing kernel used with --motion-sigma: 'gaussian' (legacy) or 'bilateral' (edge-preserving, keeps motion boundaries)")
@click.option('--motion-occlusion/--no-motion-occlusion', is_flag=True, default=False, help="occlusion-aware temporal smoothing: suppress flow carry-over where forward/backward flow is inconsistent (farneback/dis only)")
@click.option('--motion-temporal-window', type=int, default=1, help="causal spatiotemporal smoothing: average the flow over the last N frames (1 = off)")
# debugging/progress options
@click.option('--show-detector/--no-show-detector', is_flag=True, default=False, help="show the controlnet detector output")
@click.option('--show-input/--no-show-input', is_flag=True, default=False, help="show the input frame")
@click.option('--show-output/--no-show-output', is_flag=True, default=True, help="show the output frame")
@click.option('--show-motion/--no-show-motion', is_flag=True, default=False, help="show the motion transfer (not implemented yet)")
@click.option('--dump-frames', type=click.Path(), default=None, help="write intermediate frame images to a file/files during processing to visualise progress. may contain various {} placeholders")
@click.option('--skip-dumped-frames', is_flag=True, default=False, help="read dumped frames from a previous run instead of processing the input video")
@click.option('--dump-video', is_flag=True, default=False, help="write intermediate dump images to the final video instead of just the final output image")
# Color-drift fixing options
@click.option('--color-fix', type=click.Choice(['none', 'rgb', 'hsv', 'lab']), default='lab', help="prevent color from drifting due to feedback and model bias by fixing the histogram to the first frame. specify colorspace for histogram matching, e.g. 'rgb' or 'hsv' or 'lab', or 'none' to disable.")
@click.option('--color-amount', type=float, default=0.0, help="blend between the original color and the color matched version, 0.0-1.0")
@click.option('--color-info', is_flag=True, default=False, help="print extra stats about the color content of the output to help debug color drift issues")
# Detector-specific options
@click.option('--canny-low-thr', type=float, default=100, help="canny edge detector lower threshold (fluxcanny, unionxl --control-type canny)")
@click.option('--canny-high-thr', type=float, default=200, help="canny edge detector higher threshold (fluxcanny, unionxl --control-type canny)")
# reference only (--controlnet refxl)
@click.option('--swap-images', is_flag=True, default=False, help="Switch the init and reference images when using reference-only controlnet")
# video-filter chains (see videofilters.py --list-filters); each is an animatable filter-chain
# spec applied at one point in the frame loop. spec values may be animation curves evaluated on
# the same x as --audio-animate. e.g. --input-filter 'bilateral d=7; saturation amount=1.2'
@click.option('--input-filter', type=str, default=None, help="filter chain applied to each raw input frame (affects everything downstream)")
@click.option('--feedforward-filter', type=str, default=None, help="filter chain applied to the input's feedforward contribution to the next init image")
@click.option('--motion-filter', type=str, default=None, help="filter chain applied to the frames fed into the motion estimator")
@click.option('--control-filter', type=str, default=None, help="filter chain applied to the control/conditioning image")
@click.option('--feedback-filter', type=str, default=None, help="filter chain applied to the motion-warped previous output before it is fed back")
@click.option('--output-filter', type=str, default=None, help="filter chain applied to the final output frame")
def main(input_video, output_video, overwrite, start_time, end_time,
		duration, output_bitrate, output_codec,
		max_dimension, min_dimension, round_dims_to,
		no_audio, audio_from, audio_offset, audio_animate, prompt, negative_prompt,
		prompt_strength, num_inference_steps, controlnet,
		controlnet_strength, fix_orientation, init_image_strength,
		feedthrough_strength, motion_estimator, motion_alpha, motion_sigma,
		motion_spatial_filter, motion_occlusion, motion_temporal_window,
		show_detector, show_input, show_output, show_motion,
		dump_frames, skip_dumped_frames, dump_video,
		color_fix, color_amount, color_info, canny_low_thr=None,
		canny_high_thr=None,
		swap_images = False,
		input_filter=None, feedforward_filter=None, motion_filter=None,
		control_filter=None, feedback_filter=None, output_filter=None,
		control_type='depth', union_model='brad-twinkl/controlnet-union-sdxl-1.0-promax',
		ip_adapter=False, ip_adapter_scale=0.6, ip_adapter_source='feedback'
		):

	# substitute {} placeholders in output_video with input_video basename
	if output_video != None and input_video != None:
			inputpath=pathlib.Path(input_video).resolve()
			output_video = output_video.format(inpath=str(inputpath), indir=str(inputpath.parent), instem=str(inputpath.stem))
			output_video_path = pathlib.Path(output_video)
			if not output_video_path.parent.exists():
				output_video_path.parent.mkdir(parents=True)
			output_video = str(output_video_path)
			if overwrite == False:
				output_video = find_unused_filename(output_video)
			print(f"Input video: {inputpath}")
			print(F"Output video: {output_video}")
	else:
			raise ArgumentError(f"Must specify both input and output video filenames as positional arguments")

	# split audio animation argument into a dict to pass into the processing function below
	animatable_params = ["denoise", "feedthrough", "control", "guidance", "ipadapter"]
	if audio_animate != None:
		audio_params = {}
		for audio_anim_stmt in audio_animate.split(";"):
			audio_anim_assign = audio_anim_stmt.split("=", 1)
			if len(audio_anim_assign) < 2:
				raise ArgumentError(f"Invalid --audio-animate argument: missing '=' assignment in section '{audio_anim_stmt}'!")
			else:
				key, val = audio_anim_assign
				key = key.strip()
				val = val.strip()
				if key != "" and (animatable_params == None or key in animatable_params):
					if key in audio_params.keys():
						raise KeyError(f"Duplicate parameter in --audio-animate argument: '{key}' already specified once!")
					audio_params[key] = val
				else:
					raise KeyError(f"Invalid parameter in --audio-animate argument: '{key}' is not one of {repr(animatable_params)}!")
		print("Parsed {len(audio_params)} audio-animation parameter curves! Wheeeee...")
	else:
		audio_params = None

	# parse the video-filter chains once, before the slow models load, so a bad spec fails fast.
	# each is a videofilters.FilterChain applied at one point in the frame loop, or None if unset.
	def build_filter_chain(spec, label):
		if not spec:
			return None
		try:
			chain = videofilters.parse_video_filter_spec(spec)
			print(f"Parsed --{label}-filter: {len(chain.steps)} filter(s), requires_depth={chain.requires_depth}")
			return chain
		except Exception as e:
			raise ArgumentError(f"Invalid --{label}-filter chain spec: {e}")

	input_chain = build_filter_chain(input_filter, "input")
	feedforward_chain = build_filter_chain(feedforward_filter, "feedforward")
	motion_chain = build_filter_chain(motion_filter, "motion")
	control_chain = build_filter_chain(control_filter, "control")
	feedback_chain = build_filter_chain(feedback_filter, "feedback")
	output_chain = build_filter_chain(output_filter, "output")

	# run controlnet pipeline on video

	# choose the controlnet model and detector based on the --controlnet option
	use_sdxl = False
	use_flux = False
	use_sdxl_reference = False
	use_union = False
	union_control_mode = None

	print("Loading controlnet model:", controlnet)

	if controlnet == 'unionxl':
		import torch  # local, because other branches in main do `import torch` (making it function-local)
		# SDXL ControlNet-Union: one controlnet, the --control-type picks the detector + mode.
		# union control mode indices (xinsir ControlNetPlus): 0=openpose 1=depth 2=softedge 3=canny 4=normal 5=segment
		union_modes = {'openpose': 0, 'depth': 1, 'softedge': 2, 'canny': 3, 'normal': 4}
		union_control_mode = union_modes[control_type]
		if control_type == 'depth':
			detector_kwargs = dict()
			detector_model = DPTDepthDetectorWrapper()
		elif control_type == 'canny':
			detector_kwargs = dict({
				"low_threshold": canny_low_thr if canny_low_thr != None else 50,
				"high_threshold": canny_high_thr if canny_high_thr != None else 200,
			})
			detector_model = CannyDetector()
		elif control_type == 'openpose':
			from controlnet_aux import OpenposeDetector
			detector_kwargs = dict()
			detector_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
		elif control_type == 'softedge':
			from controlnet_aux import HEDdetector
			detector_kwargs = dict()
			detector_model = HEDdetector.from_pretrained("lllyasviel/ControlNet")
		elif control_type == 'normal':
			from controlnet_aux import NormalBaeDetector
			detector_kwargs = dict()
			detector_model = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
		else:
			raise NotImplementedError(f"--control-type {control_type} not supported for unionxl")
		print(f"Loading SDXL ControlNet-Union model {union_model} (control-type={control_type}, mode={union_control_mode})...")
		controlnet_model = ControlNetUnionModel.from_pretrained(union_model, torch_dtype=torch.float16)
		use_union = True
	elif controlnet == 'refxl':
		detector_kwargs = dict()
		detector_model = None #DPTDepthDetectorWrapper()
		controlnet_model = None #detector_model.load_controlnet()
		use_sdxl = False
		use_sdxl_reference = True
	elif controlnet == 'depthxl':
		detector_kwargs = dict()
		detector_model = DPTDepthDetectorWrapper()
		controlnet_model = detector_model.load_controlnet()
		use_sdxl = True
	elif controlnet == 'fluxcanny':
		from diffusers import FluxControlNetModel
		import torch
		detector_kwargs = dict({
			"low_threshold": canny_low_thr if canny_low_thr != None else 50,
			"high_threshold": canny_high_thr if canny_high_thr != None else 200
		})
		detector_model = CannyDetector()
		controlnet_model = FluxControlNetModel.from_pretrained("InstantX/FLUX.1-dev-Controlnet-Canny-alpha", torch_dtype=torch.bfloat16)
		use_flux = True
	else:
		raise NotImplementedError("controlnet type not implemented")

	# intantiate the motion estimation model (pluggable: raft / raft_small / farneback / dis)
	print(f"Loading motion estimator: {motion_estimator}...")
	motion = motionestimators.create_estimator(
		motion_estimator,
		spatial_filter=motion_spatial_filter,
		occlusion=motion_occlusion,
		temporal_window=motion_temporal_window,
	)

	# instantiate the diffusion pipeline
	if use_flux == True:
		print("Loading FLUX model...")
		from pipeline_flux_controlnet_image_to_image import FluxControlNetImg2ImgPipeline
		from transformers import T5EncoderModel
		import time
		import gc
		import torch
		import diffusers

		def flush():
		    gc.collect()
		    torch.cuda.empty_cache()

		t5_encoder = T5EncoderModel.from_pretrained(
		    "black-forest-labs/FLUX.1-schnell", subfolder="text_encoder_2", revision="refs/pr/7", torch_dtype=torch.bfloat16
		)
		text_encoder = diffusers.DiffusionPipeline.from_pretrained(
		    "black-forest-labs/FLUX.1-schnell",
		    text_encoder_2=t5_encoder,
		    transformer=None,
		    vae=None,
		    revision="refs/pr/7",
		)
		controlnet_model.to(torch.bfloat16)
		pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
		    "black-forest-labs/FLUX.1-schnell",
		    torch_dtype=torch.bfloat16,
		    revision="refs/pr/1",
		    text_encoder_2=None,
		    text_encoder=None,
		    controlnet=controlnet_model
		)
		pipe.enable_model_cpu_offload()
		#pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
		#    "black-forest-labs/FLUX.1-schnell", controlnet=controlnet_model, torch_dtype=torch.float16)
		#pipe.text_encoder.to(torch.float16)

		#pipe.enable_model_cpu_offload()
		#pipe.enable_xformers_memory_efficient_attention()
		text_encoder.to("cuda")
		start = time.time()
		(
				prompt_embeds,
				pooled_prompt_embeds,
				_,
		) = text_encoder.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)
		text_encoder.to("cpu")
		flush()
		print(f"Prompt encoding time: {time.time() - start}")
	elif use_union == True:
		print("Loading SDXL VAE...")
		import torch
		vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
		print("Loading SDXL ControlNet-Union diffusion pipeline...")
		pipe = StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
				"stabilityai/stable-diffusion-xl-base-1.0",
				controlnet=controlnet_model,
				vae=vae,
				variant="fp16",
				use_safetensors=True,
				torch_dtype=torch.float16,
		)
		if ip_adapter:
			print(f"Loading IP-Adapter (sdxl) for feedback, scale={ip_adapter_scale}...")
			pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
			pipe.set_ip_adapter_scale(ip_adapter_scale)
		pipe.enable_model_cpu_offload()
	elif use_sdxl == True:
		print("Loading SDXL VAE...")
		import torch
		vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
		print("Loading SDXL-base model diffusion pipeline...")
		pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
				"stabilityai/stable-diffusion-xl-base-1.0",
				controlnet=controlnet_model,
				vae=vae,
				variant="fp16",
				use_safetensors=True,
				torch_dtype=torch.float16,
		)
		if ip_adapter:
			print(f"Loading IP-Adapter (sdxl) for feedback, scale={ip_adapter_scale}...")
			pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
			pipe.set_ip_adapter_scale(ip_adapter_scale)
		pipe.enable_model_cpu_offload()
	elif use_sdxl_reference == True:
		import torch
		from pipeline_stable_diffusion_xl_img2img_reference import StableDiffusionXLImg2ImgReferencePipeline
		print("Loading SDXL VAE...")
		vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
		print("Loading SDXL-base model diffusion pipeline...")
		pipe = StableDiffusionXLImg2ImgReferencePipeline.from_pretrained(
				"stabilityai/stable-diffusion-xl-base-1.0",
				vae=vae,
				variant="fp16",
				use_safetensors=True,
				torch_dtype=torch.float16,
		)
		pipe.enable_model_cpu_offload()

	# ----------------------------------------------------------------------------------------------
	# okay, now we have our pipeline and auxiliary models set up, let's process the video
	# ----------------------------------------------------------------------------------------------

	print(f"Pipeline and aux models loaded successfully, starting video frame processing loop...")

	# state maintained across frames for motion estimation/transfer and output feedback loop

	first_output_frame = None
	prev_input_frame = None
	prev_motion_frame = None
	prev_output_frame = None
	prev_input_gray = None
	prev_predicted_output_frame = None
	flow = None

	def frame_filter(framenum, input_frame, _anim_x=0.0, _anim_fps=None, **params):
		# state kept between frames
		nonlocal prev_input_frame
		nonlocal prev_motion_frame
		nonlocal prev_input_gray
		nonlocal prev_output_frame
		nonlocal prev_predicted_output_frame
		nonlocal first_output_frame
		nonlocal flow

		# helper to run one of the video-filter chains at the current animation position
		def apply_chain(chain, image):
			if chain is None or image is None:
				return image
			return chain(image, x=_anim_x, n=framenum, fps=_anim_fps)

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
		# INPUT filter point: applied to the raw input frame, affects everything downstream
		input_image = apply_chain(input_chain, input_image)
		input_frame = np.asarray(input_image)

		# Converts each frame to grayscale as required by optical flow
		input_gray = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)

		output_frame = None
		skipped = False
		skipped_frame = None
		if dump_frames != None:
			inputpath=pathlib.Path(input_video).resolve()
			dump_frames_filename = dump_frames.format(inpath=str(inputpath), indir=str(inputpath.parent), instem=str(inputpath.stem), n=framenum)
			try:
				if pathlib.Path(dump_frames_filename).exists() and skip_dumped_frames:
					print(f"Skipping frame {framenum} because {dump_frames_filename} already exists, reading output image from file...")
					skipped_frame = PIL.Image.open(dump_frames_filename)
					output_frame = skipped_frame.crop((skipped_frame.width - width, 0, skipped_frame.width, height))
					skipped = True
			except Exception as e:
				print(f"Error reading {dump_frames_filename}, continuing...  {e}")
		else:
			dump_frames_filename = None

		# MOTION filter point: the (possibly separately filtered) frames fed to the estimator.
		# kept in its own prev_motion_frame stream so both frames compared are filtered alike.
		motion_input = apply_chain(motion_chain, input_image.copy()) if motion_chain is not None else input_image

		# if we have a previous frame, transfer motion from the input to the output and blend with the noise
		if prev_motion_frame is not None and not skipped:
			#flow, smoothed_flow = estimate_motion_raft(prev_input_frame, input_frame, flow, motion_smoothing)
			flow = motion.estimate_motion(prev_motion_frame, motion_input, flow, motion_alpha, motion_sigma)
			predicted_output_frame = motion.transfer_motion(flow, prev_output_frame)
			# FEEDBACK filter point: the motion-warped previous output, before it is fed back
			predicted_output_frame = apply_chain(feedback_chain, predicted_output_frame)
		else:
			predicted_output_frame = None

		if not skipped:
			if detector_model != None:
				control_image = detector_model(input_image, **detector_kwargs).convert("RGB")
			else:
				control_image = input_image.copy()
			control_image = control_image.resize((input_image.width, input_image.height))
			# CONTROL filter point: applied to the control/conditioning image fed to the pipeline
			control_image = apply_chain(control_chain, control_image)
			ci = np.asarray(control_image)
			#print(f"ci.shape={ci.shape} ci.min={ci.min()} ci.max={ci.max()} ci.mean={ci.mean()} ci.std={ci.std()}")

		print(f"*** Params for this frame: \n*** {repr(params)}")
		if "feedthrough" in params.keys():
			feedthrough_strength = params["feedthrough"]
		if "denoise" in params.keys():
			init_image_strength = params["denoise"]
		if "control" in params.keys():
			controlnet_strength = params["control"]
		if "guidance" in params.keys():
			prompt_strength = params["guidance"]
		if ip_adapter and "ipadapter" in params.keys():
			# IP-Adapter strength is animatable; apply the per-frame scale to the loaded adapter
			pipe.set_ip_adapter_scale(params["ipadapter"])

		# IP-Adapter feedback reference: the image used as the image-prompt for this frame.
		# 'feedback' = motion-warped previous output (falls back to the input on the first frame).
		if ip_adapter:
			if ip_adapter_source == 'input':
				ip_ref = input_image
			elif ip_adapter_source == 'output':
				ip_ref = prev_output_frame if prev_output_frame is not None else input_image
			else:  # 'feedback'
				ip_ref = predicted_output_frame if predicted_output_frame is not None else input_image
		else:
			ip_ref = None

		# FEEDFORWARD filter point: the input's contribution to the next init image
		feedforward_image = apply_chain(feedforward_chain, input_image.copy()) if feedforward_chain is not None else input_image

		if predicted_output_frame is None:
			strength = feedthrough_strength * init_image_strength
			init_image = feedforward_image.copy()
		else:
			strength = init_image_strength
			init_image = PIL.Image.blend(feedforward_image, predicted_output_frame, 1.0 - feedthrough_strength)

		if output_frame == None:
			ii = np.asarray(init_image)
			#print(f"ii.shape={ii.shape} ii.min={ii.min()} ii.max={ii.max()} ii.mean={ii.mean()} ii.std={ii.std()}")

			if use_flux == True:
			    output_frame = pipe(
			        prompt_embeds=prompt_embeds.bfloat16(),
			        pooled_prompt_embeds=pooled_prompt_embeds.bfloat16(),
			        width=width,
			        height=height,
							image=init_image,
							control_image=control_image,
							controlnet_conditioning_scale=controlnet_strength,
							strength=1.0-strength,
							num_inference_steps=num_inference_steps,
							guidance_scale=(prompt_strength/7.0)*3.5,
					).images[0]
			elif use_union == True:
					union_kwargs = dict(
						prompt=prompt,
						guidance_scale=prompt_strength,
						negative_prompt=negative_prompt,
						num_inference_steps=num_inference_steps,
						generator=generator,
						image=init_image,
						strength=1.0 - strength,
						control_image=[control_image],
						control_mode=[union_control_mode],
						controlnet_conditioning_scale=controlnet_strength,
						width=width,
						height=height,
					)
					if ip_adapter:
						union_kwargs["ip_adapter_image"] = ip_ref
					output_frame = pipe(**union_kwargs).images[0]
			elif use_sdxl == True:
					sdxl_kwargs = dict(
						prompt=prompt,
						guidance_scale=prompt_strength,
						negative_prompt=negative_prompt,
						num_inference_steps=num_inference_steps,
						generator=generator,
						control_image=control_image,
						controlnet_conditioning_scale=controlnet_strength,
						image=init_image,
						strength=1.0 - strength,
					)
					if ip_adapter:
						sdxl_kwargs["ip_adapter_image"] = ip_ref
					output_frame = pipe(**sdxl_kwargs).images[0]
			elif use_sdxl_reference == True:
					output_frame = pipe(
						prompt=prompt,
						guidance_scale=prompt_strength,
						negative_prompt=negative_prompt,
						num_inference_steps=num_inference_steps,
						generator=generator,
						ref_image = control_image if swap_images else init_image,
						#controlnet_conditioning_scale=controlnet_strength,
						image = init_image if swap_images else control_image,
						strength = 1.0 - strength,
					).images[0]

		if color_fix != 'none' and color_amount > 0.0:
			if first_output_frame == None:
				# save the first output frame for color correction
				first_output_frame = output_frame.copy()
			else:
				# skipped frames don't get color correction, since they already have it applied
				if not skipped:
					# blend the color fix into the output frame
					output_frame = PIL.Image.fromarray((
						np.asarray(output_frame)*(1.0-color_amount) +
						maintain_colors(np.asarray(first_output_frame), np.asarray(output_frame), color_fix)*color_amount
					).astype(np.uint8))

		# OUTPUT filter point: applied to the final output frame (and so to what is fed back).
		# skipped frames are read back already-filtered from a previous dump, so don't re-apply.
		if not skipped:
			output_frame = apply_chain(output_chain, output_frame)

		if show_motion:
			if not (flow is None):
				motion_image = motion.flow_to_image(flow)
			else:
				motion_image = PIL.Image.new("RGB", (width, height), (0,0,0))

		final_frame = output_frame
		if (dump_frames != None or dump_video):
			if skipped and dump_video:
				final_frame = skipped_frame
			elif not skipped:
				final_frame = stackh( list(
					list([input_image] if show_input else []) +
					list([motion_image] if show_motion else []) +
					list([control_image] if show_detector else []) +
					list([output_frame] if show_output else [])) )

		if color_info:
			meanhslout = np.mean(rgbtohsl(np.asarray(output_frame)), axis = (0,1))
			print(f"output color info: mean hue{meanhslout[0]}, mean sat={meanhslout[1]}, mean lum={meanhslout[2]}")
			if not (prev_output_frame is None):
				prevmeanhslout = np.mean(rgbtohsl(np.asarray(prev_output_frame)), axis = (0,1))
				diffhslout = meanhslout - prevmeanhslout
				print(f"output color diff: mean hue{diffhslout[0]}, mean sat={diffhslout[1]}, mean lum={diffhslout[2]}")

		if dump_frames != None and not skipped:
			# inputpath=pathlib.Path(input_video).resolve()
			# dump_frames_filename = dump_frames.format(inpath=str(inputpath), indir=str(inputpath.parent), instem=str(inputpath.stem), n=framenum)
			print(f"Dumping frame {framenum} as png to {dump_frames_filename}...")
			if pathlib.Path(dump_frames_filename).parent.is_dir() == False:
				pathlib.Path(dump_frames_filename).parent.mkdir(parents=True, exist_ok=True)
			final_frame.save(dump_frames_filename)

		#output_frame.save("frame.png")
		prev_input_frame = input_image.copy()
		prev_motion_frame = motion_input.copy()
		prev_input_gray = input_gray
		prev_output_frame = output_frame.copy()
		prev_predicted_output_frame = predicted_output_frame
		if dump_video:
			return final_frame
		else:
			return output_frame

	process_frames(input_video, audio_from, output_video, frame_filter, start_time, end_time, duration,
		None, # rescale_time
		audio_params,
		max_dimension, min_dimension, round_dims_to, fix_orientation, output_bitrate, output_codec)

if __name__ == "__main__":
	main()
