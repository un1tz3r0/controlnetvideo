"""
!pip install -q transformers xformers accelerate git+https://github.com/huggingface/accelerate.git
!pip install -q git+https://github.com/mikegarts/diffusers.git@stablediffusion.controlnet.img2img.pipeline
"""

import click
import numpy as np
import torch
import PIL.Image
import cv2
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler
from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from typing import Tuple, List, FrozenSet, Sequence, MutableSequence, Mapping, Optional, Any, Type, Union
from controlnet_aux import CannyDetector, OpenposeDetector, MLSDdetector, HEDdetector, MidasDetector
import pathlib


# ----------------------------------------------------------------------
# Motion estimation using the RAFT optical flow model (and some legacy
# farneback code that is not currently used)
# ----------------------------------------------------------------------

from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
import torchvision.transforms.functional
import torch

class RAFTMotionEstimator:
	def __init__(self):
		self.raft_weights = Raft_Large_Weights.DEFAULT
		self.raft_transforms = self.raft_weights.transforms()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
		self.raft_model = self.raft_model.eval()

	def estimate_motion(self, srcframe1, srcframe2, prev_flow:Optional[np.ndarray] = None, alpha=0.0, sigma=0.0):
		'''
		estimate dense optical flow from srcframe1 to srcframe2, using the Raft model
		
		exponential temporal smoothing is applied using the previous flow, according to 
		alpha, which should be between 0.0 and 1.0.  If alpha is 0.0, no temporal smoothing
		is applied. the flow will be smoothed spatially using a gaussian filter with sigma
		if sigma > 0.0.  If sigma is 0.0, no spatial smoothing is applied.
		'''
		
		# put the two frames into tensors
		img1_batch = torch.tensor(np.asarray(srcframe1).transpose((2, 1, 0))).unsqueeze(0)
		img2_batch = torch.tensor(np.asarray(srcframe2).transpose((2, 1, 0))).unsqueeze(0)

		# resize the frames to multiples of 8 for the RAFT model (which requires this)
		img1_batch = torchvision.transforms.functional.resize(img1_batch, size=[(srcframe1.width//8)*8, (srcframe1.height//8)*8], antialias=False)
		img2_batch = torchvision.transforms.functional.resize(img2_batch, size=[(srcframe2.width//8)*8, (srcframe2.height//8)*8], antialias=False)
		
		# apply the transforms required by the RAFT model
		img1_batch, img2_batch = self.raft_transforms(img1_batch, img2_batch, )

		# Compute the dense optical flow from frame1 to frame2
		list_of_flows = self.raft_model(img1_batch.to(self.device), img2_batch.to(self.device))
		flow = list_of_flows[0].cpu().detach().numpy()
		
		if len(flow.shape) == 4:
			if flow.shape[0] == 1:
				flow = flow[0]
		if flow.shape[0] == 2:
			flow = np.transpose(flow, (2, 1, 0))

		# smooth the flow spatially using a gaussian filter if sigma > 0.0
		#if sigma > 0.0:
		#	flow = cv2.bilateralFilter(flow, 9, sigma, sigma)
		if sigma > 0.0:
			flow = cv2.GaussianBlur(flow, (0,0), sigma)
		# else:
			#flow = flow[0]
			#flow[...,0] = cv2.GaussianBlur(flow[...,0], (0, 0), sigma)
			#flow[...,1] = cv2.GaussianBlur(flow[...,1], (0, 0), sigma)

		# smooth the flow using exponential temporal smoothing if alpha is < 1.0
		if not (prev_flow is None) and alpha > 0.0:
			flow = prev_flow * alpha + flow * (1.0 - alpha)
		
		# return the smoothed flow
		return flow

	def transfer_motion(self, flow, destframe, reverse_order=True):
		'''
		transfer motion from dense optical flow onto destframe^t-1 to get destframe^t

		reverse order is used when the y axis is the first plane of the flow, and the x is the second,
		as in the results from raft
		'''
		if len(flow.shape) == 4:
			flow = np.transpose(flow[0], (2, 1, 0))
		
		# Compute the inverse flow from frame2 to frame1
		inv_flow = -flow
		destframe = np.asarray(destframe)
		
		# Warp frame3 using the inverse flow
		h, w = destframe.shape[:2]
		x, y = np.meshgrid(np.arange(w), np.arange(h))
		x_inv = np.round(x + inv_flow[...,0 if not reverse_order else 1]).astype(np.float32)
		y_inv = np.round(y + inv_flow[...,1 if not reverse_order else 0]).astype(np.float32)
		x_inv = np.clip(x_inv, 0, w-1)
		y_inv = np.clip(y_inv, 0, h-1)
		warped = cv2.remap(np.asarray(destframe), x_inv, y_inv, cv2.INTER_LINEAR)

		return PIL.Image.fromarray(warped)

	def flow_to_image(self, flow):
		if len(flow.shape) == 4:
			flow = np.transpose(flow[0], (2, 1, 0))
		''' convert dense optical flow to image, direction -> hue, magnitude -> brightness '''
		hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
		hsv[...,1] = 255
		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		hsv[...,0] = ang*180/np.pi/2
		hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
		return PIL.Image.fromarray(rgb)

"""
# not currently used...

def extract_first_moment_from_flow(flow):
	''' extract first moment from dense optical flow '''
	moment = np.mean(flow, axis=(0,1))
	flow = np.array(flow[:,:,:] - moment[None, None, :])
	return flow, moment

def estimate_motion_farneback(srcframe1, srcframe2, prev_flow:Optional[np.ndarray] = None, alpha=1.0, sigma=50.0):
	''' given the current frame and the prior frame, estimate dense optical flow
	for the motion between them. if the prior frame's estimated flow is provided,
	use it as a hint for the current frame's flow. '''

	# Convert the frames to grayscale
	gray1 = cv2.cvtColor(np.asarray(srcframe1), cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(np.asarray(srcframe2), cv2.COLOR_BGR2GRAY)

	# Compute the dense optical flow from frame1 to frame2
	if prev_flow is None:
		flow2 = None
	else:
		flow2 = prev_flow.copy() # maybe copy is unnescessary

	flow = cv2.calcOpticalFlowFarneback(gray1, gray2, flow2,
			0.33, # pyr_scale: parameter, specifying the image scale (&lt;1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
			2, # levels: number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
			30, # winsize: averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
			5, # iterations: number of iterations the algorithm does at each pyramid level.
			13, # poly_n: size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
			2.7, # poly_sigma: standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
			0 if flow2 is None else cv2.OPTFLOW_USE_INITIAL_FLOW)

	if not (prev_flow is None):
		smoothed_flow = prev_flow * (1-alpha) + flow * (alpha)
	else:
		smoothed_flow = flow
	smoothed_flow = cv2.GaussianBlur(smoothed_flow, (0,0), sigma)

	''' returns a tuple of (flow, smoothed_flow) where flow is the raw flow and smoothed_flow is
	spatial and temporally smoothed flow. the smoothed flow should be used for motion transfer,
	while the raw flow should be used for motion estimation on the next frame. '''
	return flow, smoothed_flow

def flow_to_image_farneback(flow):
	''' convert dense optical flow to image, direction -> hue, magnitude -> brightness '''
	hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
	hsv[...,1] = 255
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	return PIL.Image.fromarray(rgb)
"""

## never got this working yet, could significantly improve results
#
# def smooth_flow_spatiotemporal(flow, sigma:float):
# 	''' smooth dense optical flow using a 3D edge-preserving filter. note implementing this
# 	with opencv is quite possible but would require delaying the frame output due to the
# 	nonlocal-means denoising algorithm's working on the middle frame of a temporal window
# 	consisting of an odd number of frames so as to be symmetrical in lookahead and previously
# 	seen frames. '''
#
# 	cv2.fastNlMeansDenoisingMulti(flow, flow, sigma, 0, 0)

# -----------------------------------------------------------------------------------------------
# helpers for depth-based controlnets
# -----------------------------------------------------------------------------------------------

class MidasDetectorWrapper:
	''' a wrapper around the midas detector model which allows
	choosing either the depth or the normal output on creation '''
	def __init__(self, output_index=0, **kwargs):
		self.model = MidasDetector()
		self.output_index = output_index
		self.default_kwargs = dict(kwargs)
	def __call__(self, image, **kwargs):
		ka = dict(list(self.default_kwargs.items()) + list(kwargs.items()))
		#return torch.tensor(self.model(np.asarray(image), **ka)[self.output_index][None, :, :].repeat(3,0)).unsqueeze(0)
		return PIL.Image.fromarray(self.model(np.asarray(image), **ka)[self.output_index]).convert("RGB")


def depth_to_normal(image):
		''' converts 2d 1ch (z) grayscale depth map image
							to 2d 3ch (xyz) surface normal map image using sobel filter and cross product '''
		image_depth = image.copy()
		image_depth -= np.min(image_depth)
		image_depth /= np.max(image_depth)

		bg_threshold = 0.1

		x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
		x[image_depth < bg_threshold] = 0

		y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
		y[image_depth < bg_threshold] = 0

		z = np.ones_like(x) * np.pi * 2.0

		image = np.stack([x, y, z], axis=2)
		image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
		image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
		image = Image.fromarray(image)
		return image

# -----------------------------------
# general image helpers
# -----------------------------------

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

def hsltorgb(hsl:np.ndarray):
	''' vectorized hsl to rgb conversion 
	input is a numpy array of shape (..., 3) and dtype float, with hue first in 0-360, then sat and lum in 0-1 '''
	h, s, l = hsl[:,:,0], hsl[:,:,1], hsl[:,:,2]
	c = (1 - np.abs(2*l-1)) * s
	h = h / 60
	x = c * (1 - np.abs(h % 2 - 1))
	m = l - c/2
	r = np.zeros(h.shape)
	g = np.zeros(h.shape)
	b = np.zeros(h.shape)
	r[h < 1] = c[h < 1]
	r[h >= 1] = x[h >= 1]
	g[h < 1] = x[h < 1]
	g[h >= 2] = c[h >= 2]
	b[h < 2] = c[h < 2]
	b[h >= 3] = x[h >= 3]
	r[h >= 4] = c[h >= 4]
	g[h >= 4] = x[h >= 4]
	r += m
	g += m
	b += m
	r *= 255
	g *= 255
	b *= 255
	return np.ndarray([r, g, b])

def hsltorgb(hsl:np.ndarray):
  h, s, l = hsl[:,:,0], hsl[:,:,1], hsl[:,:,2]
  c = (1 - np.abs(2*l-1)) * s
  h = h / 60
  x = c * (1 - np.abs(h % 2 - 1))
  m = l - c/2
  r = np.zeros(h.shape)
  g = np.zeros(h.shape)
  b = np.zeros(h.shape)
  r[h < 1] = c[h < 1]
  r[h >= 1] = x[h >= 1]
  g[h < 1] = x[h < 1]
  g[h >= 2] = c[h >= 2]
  b[h < 2] = c[h < 2]
  b[h >= 3] = x[h >= 3]
  r[h >= 4] = c[h >= 4]
  g[h >= 4] = x[h >= 4]
  r += m
  g += m
  b += m
  r *= 255
  g *= 255
  b *= 255
  return np.ndarray([r, g, b])

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
import cv2

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

def process_frames(input_video, output_video, wrapped, start_time=None, end_time=None, duration=None, max_dimension=None, min_dimension=None, round_dims_to=None, fix_orientation=False, ):
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
		from time import monotonic
		try:
			framenum = 0
			starttime = monotonic()
			def wrapper(gf, t):
				nonlocal framenum
				nonlocal starttime
				elapsed = monotonic() - starttime
				if t > 0:
					eta = (video.duration / t) * elapsed
				else:
					eta = 0
				print(f"Processing frame {framenum} at time {t}/{video.duration} seconds... {elapsed:.2f}s elapsed, {eta:.2f}s estimated time remaining")
				result = wrapped(framenum, PIL.Image.fromarray(gf(t)).resize((w,h)))
				framenum = framenum + 1
				return np.asarray(result)

			#video.fx(wrapper).write_videofile(output_video)
			video.fl(wrapper, keep_duration=True).write_videofile(output_video)
			#processed_video = mp.ImageSequenceClip([
			#  np.array(wrapped(framenum, Image.fromarray(frame).resize((w,h))))
			#    for framenum, frame in
			#      enumerate(tqdm(video.iter_frames()))
			#  ], fps=video.fps)
		finally:
			# save the video
			#if processed_video != None:
			#  processed_video.write_videofile(output_video)
			video.close()

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.option('--start-time', type=float, default=None, help="start time in seconds")
@click.option('--end-time', type=float, default=None, help="end time in seconds")
@click.option('--duration', type=float, default=None, help="duration in seconds")
@click.option('--max-dimension', type=int, default=832, help="maximum dimension of the video")
@click.option('--min-dimension', type=int, default=512, help="minimum dimension of the video")
@click.option('--round-dims-to', type=int, default=128, help="round the dimensions to the nearest multiple of this number")
# not implemented... yet:
@click.option('--no-audio', is_flag=True, default=False, help="don't include audio in the output video, even if the input video has audio")
@click.option('--audio-from', type=click.Path(exists=True), default=None, help="audio file to use for the output video, replaces the audio from the input video, will be truncated to duration of input or --duration if given")
@click.option('--audio-offset', type=float, default=None, help="offset in seconds to start the audio from, when used with --audio-from")
# stable diffusion options
@click.option('--prompt', type=str, default=None, help="prompt used to guide the denoising process")
@click.option('--negative-prompt', type=str, default=None, help="negative prompt, can be used to prevent the model from generating certain words")
@click.option('--prompt-strength', type=float, default=7.5, help="how much influence the prompt has on the output")
#@click.option('--scheduler', type=click.Choice(['default']), default='default', help="which scheduler to use")
@click.option('--num-inference-steps', '--steps', type=int, default=25, help="number of inference steps, depends on the scheduler, trades off speed for quality. 20-50 is a good range from fastest to best.")
@click.option('--controlnet', type=click.Choice(['aesthetic', 'lineart21', 'hed', 'hed21', 'canny', 'canny21', 'openpose', 'openpose21', 'depth', 'depth21', 'normal', 'mlsd']), default='hed', help="which pretrained controlnet annotator to use")
@click.option('--controlnet-strength', type=float, default=1.0, help="how much influence the controlnet annotator's output is used to guide the denoising process")
@click.option('--fix-orientation/--no-fix-orientation', is_flag=True, default=True, help="resize videos shot in portrait mode on some devices to fix incorrect aspect ratio bug")
@click.option('--init-image-strength', type=float, default=0.5, help="the init-image strength, or how much of the prompt-guided denoising process to skip in favor of starting with an existing image")
@click.option('--feedthrough-strength', type=float, default=0.0, help="the ratio of input to motion compensated prior output to feed through to the next frame")
@click.option('--motion-alpha', type=float, default=0.1, help="smooth the motion vectors over time, 0.0 is no smoothing, 1.0 is maximum smoothing")
@click.option('--motion-sigma', type=float, default=0.3, help="smooth the motion estimate spatially, 0.0 is no smoothing, used as sigma for gaussian blur")
@click.option('--show-detector/--no-show-detector', is_flag=True, default=False, help="show the controlnet detector output")
@click.option('--show-input/--no-show-input', is_flag=True, default=False, help="show the input frame")
@click.option('--show-output/--no-show-output', is_flag=True, default=True, help="show the output frame")
@click.option('--show-motion/--no-show-motion', is_flag=True, default=False, help="show the motion transfer (not implemented yet)")
@click.option('--dump-frames', type=click.Path(), default=None, help="write intermediate frame images to a file/files during processing to visualise progress. may contain various {} placeholders")
@click.option('--skip-dumped-frames', is_flag=True, default=False, help="read dumped frames from a previous run instead of processing the input video")
@click.option('--dump-video', is_flag=True, default=False, help="write intermediate dump images to the final video instead of just the final output image")
@click.option('--color-fix', type=click.Choice(['none', 'rgb', 'hsv', 'lab']), default='lab', help="prevent color from drifting due to feedback and model bias by fixing the histogram to the first frame. specify colorspace for histogram matching, e.g. 'rgb' or 'hsv' or 'lab', or 'none' to disable.")
@click.option('--color-amount', type=float, default=0.0, help="blend between the original color and the color matched version, 0.0-1.0")
@click.option('--color-info', is_flag=True, default=False, help="print extra stats about the color content of the output to help debug color drift issues")
@click.option('--canny-low-thr', type=float, default=100, help="canny edge detector lower threshold")
@click.option('--canny-high-thr', type=float, default=200, help="canny edge detector higher threshold")
@click.option('--mlsd-score-thr', type=float, default=0.1, help="mlsd line detector v threshold")
@click.option('--mlsd-dist-thr', type=float, default=0.1, help="mlsd line detector d threshold")
def main(input_video, output_video, start_time, end_time, 
	 	duration, max_dimension, min_dimension, round_dims_to, 
		no_audio, audio_from, audio_offset, prompt, negative_prompt, 
		prompt_strength, num_inference_steps, controlnet, 
		controlnet_strength, fix_orientation, init_image_strength, 
		feedthrough_strength, motion_alpha, motion_sigma, 
		show_detector, show_input, show_output, show_motion, 
		dump_frames, skip_dumped_frames, dump_video, 
		color_fix, color_amount, color_info, canny_low_thr=None, 
		canny_high_thr=None, mlsd_score_thr=None, mlsd_dist_thr=None
	):
	
	# substitute {} placeholders in output_video with input_video basename
	if output_video != None and input_video != None:
			inputpath=pathlib.Path(input_video).resolve()
			output_video = output_video.format(inpath=str(inputpath), indir=str(inputpath.parent), instem=str(inputpath.stem))
			output_video_path = pathlib.Path(output_video)
			if not output_video_path.parent.exists():
				output_video_path.parent.mkdir(parents=True)
			output_video = str(output_video_path)
	
	# run controlnet pipeline on video

	# choose controlnet model and detector based on the --controlnet option
	# this also affects the default scheduler and the stable diffusion model version required, in the case of aesthetic controlnet
	scheduler = 'unipc'
	sdmodel = 'runwayml/stable-diffusion-v1-5'

	if controlnet == 'canny':
		detector_kwargs = dict({
			"low_threshold": canny_low_thr if canny_low_thr != None else 50,
			"high_threshold": canny_high_thr if canny_high_thr != None else 200
		})
		detector_model = CannyDetector()
		controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
	elif controlnet == 'canny21':
		detector_kwargs = dict({
			"low_threshold": canny_low_thr if canny_low_thr != None else 50,
			"high_threshold": canny_high_thr if canny_high_thr != None else 200
		})
		detector_model = CannyDetector()
		controlnet_model = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-canny-diffusers", torch_dtype=torch.float16)
		sdmodel = 'stabilityai/stable-diffusion-2-1'
	elif controlnet == 'openpose':
		detector_kwargs = dict()
		detector_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
		controlnet_model = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16)
	elif controlnet == 'openpose21':
		detector_kwargs = dict()
		detector_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
		controlnet_model = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-openpose-diffusers", torch_dtype=torch.float16)
		sdmodel = 'stabilityai/stable-diffusion-2-1'
	elif controlnet == 'depth':
		detector_kwargs = dict()
		detector_model = MidasDetectorWrapper()
		controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
	elif controlnet == 'depth21':
		detector_kwargs = dict()
		detector_model = MidasDetectorWrapper()
		controlnet_model = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-depth-diffusers", torch_dtype=torch.float16)
		sdmodel = 'stabilityai/stable-diffusion-2-1'
	elif controlnet == 'mlsd':
		detector_kwargs = dict({
			'thr_v': 0.1 if mlsd_score_thr==None else mlsd_score_thr,
			'thr_d': 0.1 if mlsd_dist_thr==None else mlsd_dist_thr
		})
		detector_model = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
		controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16)
	elif controlnet == 'lineart21':
		detector_kwargs = dict({
			'thr_v': 0.1 if mlsd_score_thr==None else mlsd_score_thr,
			'thr_d': 0.1 if mlsd_dist_thr==None else mlsd_dist_thr
		})
		detector_model = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
		controlnet_model = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-lineart-diffusers", torch_dtype=torch.float16)
		sdmodel = 'stabilityai/stable-diffusion-2-1'
	elif controlnet == 'hed':
		detector_kwargs = dict()
		detector_model = HEDdetector.from_pretrained("lllyasviel/ControlNet")
		controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
	elif controlnet == 'hed21':
		detector_kwargs = dict()
		detector_model = HEDdetector.from_pretrained("lllyasviel/ControlNet")
		controlnet_model = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-hed-diffusers", torch_dtype=torch.float16)
		sdmodel = 'stabilityai/stable-diffusion-2-1'
	elif controlnet == 'normal':
		detector_kwargs = dict()
		detector_model = MidasDetectorWrapper(1)
		controlnet_model = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16)
	elif controlnet == 'aesthetic':
		detector_kwargs = dict({
			"low_threshold": canny_low_thr if canny_low_thr != None else 50,
			"high_threshold": canny_high_thr if canny_high_thr != None else 200
		})
		detector_model = CannyDetector()
		controlnet_model = None
		sdmodel = "krea/aesthetic-controlnet"
	else:
		raise NotImplementedError("controlnet type not implemented")

	# intantiate the motion estimation model
	motion = RAFTMotionEstimator()

	# instantiate the diffusion pipeline
	if controlnet_model != None:
		pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(sdmodel, controlnet=controlnet_model, torch_dtype=torch.float16)
	else:
		pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(sdmodel, torch_dtype=torch.float16)

	# set the scheduler... this is a bit hacky but it works
	if scheduler == 'unipcm':
		pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
	elif scheduler == 'eulera':
		pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


	pipe.enable_xformers_memory_efficient_attention()
	pipe.enable_model_cpu_offload()
	pipe.run_safety_checker = lambda image, text_embeds, text_embeds_negative: (image, False)

	first_output_frame = None
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
		nonlocal first_output_frame
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
		
		# if we have a previous frame, transfer motion from the input to the output and blend with the noise
		if prev_input_frame != None and not skipped:
			#flow, smoothed_flow = estimate_motion_raft(prev_input_frame, input_frame, flow, motion_smoothing)
			flow = motion.estimate_motion(prev_input_frame, input_image, flow, motion_alpha, motion_sigma)
			predicted_output_frame = motion.transfer_motion(flow, prev_output_frame)
		else:
			predicted_output_frame = None

		if not skipped:
			control_image = detector_model(input_image, **detector_kwargs).convert("RGB")
			ci = np.asarray(control_image)
			#print(f"ci.shape={ci.shape} ci.min={ci.min()} ci.max={ci.max()} ci.mean={ci.mean()} ci.std={ci.std()}")

		if predicted_output_frame == None:
			strength = feedthrough_strength * init_image_strength
			init_image = input_image.copy()
		else:
			strength = init_image_strength
			init_image = PIL.Image.blend(input_image, predicted_output_frame, 1.0 - feedthrough_strength)
			
		if output_frame == None:
			# run the pipeline
			output_frame = pipe(
				prompt=prompt,
				guidance_scale=prompt_strength,
				negative_prompt=negative_prompt,
				num_inference_steps=num_inference_steps,
				generator=generator,
				controlnet_conditioning_image=[control_image],
				controlnet_conditioning_scale=controlnet_strength,
				image = init_image,
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
		prev_input_gray = input_gray
		prev_output_frame = output_frame.copy()
		prev_predicted_output_frame = predicted_output_frame
		if dump_video:
			return final_frame
		else:
			return output_frame

	process_frames(input_video, output_video, frame_filter, start_time, end_time, duration, max_dimension, min_dimension, round_dims_to, fix_orientation)

if __name__ == "__main__":
	main()
