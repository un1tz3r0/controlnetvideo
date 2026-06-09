"""
---------------------------------------------------------------------------------------------
motionestimators.py

Pluggable dense optical-flow motion estimation for controlnetvideo, with shared flow
smoothing/denoising. Estimators all expose the same small interface:

    flow = est.estimate_motion(prev_pil, cur_pil, prev_flow=None, alpha=0.0, sigma=0.0)
    warped = est.transfer_motion(flow, frame_pil)      # motion-compensate a frame
    img = est.flow_to_image(flow)                      # hsv visualisation

Available estimators (via create_estimator(name)):
    raft / raft_large   torchvision RAFT (GPU) - the original, kept byte-for-byte
    raft_small          torchvision RAFT-small (faster, lighter)
    farneback           cv2 Farneback (CPU, no extra deps)
    dis                 cv2 DIS optical flow (CPU, fast + decent quality)

Smoothing (configured at construction, applied by the base estimate_motion):
    spatial_filter:  'gaussian' (default) or 'bilateral' (edge-preserving, per-channel)
    temporal:        exponential blend with prev_flow via `alpha` (as before), optionally
                     occlusion-aware so carry-over is suppressed where the forward/backward
                     flow is inconsistent
    temporal_window: causal spatiotemporal averaging over the last N flows (N=1 disables)

NOTE on conventions: the cv2 estimators produce standard flow (flow[...,0]=dx, [...,1]=dy).
RAFT keeps its original (axis-swapped) convention and overrides transfer/flow_to_image
accordingly; occlusion estimation (which warps the flow field) is therefore only enabled for
the standard-convention estimators.

    by Victor Condino <un1tz3r0@gmail.com> - 2026
---------------------------------------------------------------------------------------------
"""

from typing import Optional
import numpy as np
import cv2
import PIL.Image

import torch
import torchvision.transforms.functional
from torchvision.models.optical_flow import (
	raft_large, Raft_Large_Weights, raft_small, Raft_Small_Weights,
)


# ---------------------------------------------------------------------------------------------
# shared flow operations (standard cv2 convention: flow[...,0]=dx, flow[...,1]=dy)
# ---------------------------------------------------------------------------------------------

def warp_by_flow(image, flow, border=cv2.BORDER_REPLICATE):
	''' resample `image` at positions displaced by `flow` (standard convention). '''
	h, w = image.shape[:2]
	xx, yy = np.meshgrid(np.arange(w), np.arange(h))
	map_x = (xx + flow[..., 0]).astype(np.float32)
	map_y = (yy + flow[..., 1]).astype(np.float32)
	return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=border)


def smooth_spatial(flow, sigma, method='gaussian'):
	''' spatially smooth a flow field. 'gaussian' = plain blur (legacy behaviour);
	'bilateral' = edge-preserving per-channel bilateral (keeps motion boundaries crisp). '''
	if sigma is None or sigma <= 0.0:
		return flow
	flow = flow.astype(np.float32)
	if method == 'bilateral':
		d = max(3, (int(round(sigma * 2)) | 1))  # odd diameter
		sigma_color = float(sigma * 4.0)
		sigma_space = float(sigma)
		out = np.empty_like(flow)
		for c in range(flow.shape[2]):
			out[..., c] = cv2.bilateralFilter(np.ascontiguousarray(flow[..., c]), d, sigma_color, sigma_space)
		return out
	# default: gaussian (matches the original RAFT smoothing)
	return cv2.GaussianBlur(flow, (0, 0), sigma)


def smooth_temporal(flow, prev_flow, alpha, occlusion=None):
	''' exponential temporal blend: prev_flow*alpha + flow*(1-alpha). If an occlusion mask
	(1=occluded) is given, carry-over is suppressed there (trust the current flow). '''
	if prev_flow is None or alpha is None or alpha <= 0.0:
		return flow
	if prev_flow.shape != flow.shape:
		return flow
	a = float(alpha)
	if occlusion is not None:
		a = a * (1.0 - occlusion)[..., None]
	return prev_flow * a + flow * (1.0 - a)


def forward_backward_occlusion(fwd, bwd, thresh=1.5):
	''' occlusion mask (1=occluded) from forward/backward flow consistency. Uses the standard
	adaptive criterion (Sundaram et al.): a pixel is occluded where the round-trip error
	exceeds 0.01*(|fwd|^2+|bwd|^2) + thresh^2, so the tolerance grows with flow magnitude
	instead of flagging every fast-moving pixel. `thresh` is the constant error floor in px. '''
	warped_bwd = warp_by_flow(bwd, fwd)
	diff = fwd + warped_bwd
	err2 = (diff * diff).sum(axis=2)
	mag2 = (fwd * fwd).sum(axis=2) + (warped_bwd * warped_bwd).sum(axis=2)
	return (err2 > (0.01 * mag2 + thresh * thresh)).astype(np.float32)


def _time_weights(n):
	''' gaussian-in-time weights for a causal window, newest (last) sample weighted highest. '''
	dist = np.arange(n)[::-1].astype(np.float64)  # 0 = newest, larger = older
	s = max(1.0, n / 2.0)
	w = np.exp(-(dist ** 2) / (2.0 * s * s))
	return w / w.sum()


# ---------------------------------------------------------------------------------------------
# base estimator: owns smoothing; subclasses implement _estimate_raw()
# ---------------------------------------------------------------------------------------------

class MotionEstimator:
	# cv2-convention estimators set this True; RAFT (axis-swapped) leaves it False
	standard_convention = True

	def __init__(self, spatial_filter='gaussian', occlusion=False, occlusion_thresh=1.5,
	             temporal_window=1, **kwargs):
		self.spatial_filter = spatial_filter
		self.occlusion = bool(occlusion)
		self.occlusion_thresh = float(occlusion_thresh)
		self.temporal_window = max(1, int(temporal_window))
		self._hist = []
		self._last_occlusion = None
		self._warned_occ = False

	def _estimate_raw(self, prev_pil, cur_pil):
		''' return the raw dense flow from prev->cur (subclass-specific convention). '''
		raise NotImplementedError

	def estimate_motion(self, prev_pil, cur_pil, prev_flow=None, alpha=0.0, sigma=0.0):
		flow = self._estimate_raw(prev_pil, cur_pil)
		flow = smooth_spatial(flow, sigma, self.spatial_filter)

		occ = None
		if self.occlusion:
			if self.standard_convention:
				bwd = self._estimate_raw(cur_pil, prev_pil)
				bwd = smooth_spatial(bwd, sigma, self.spatial_filter)
				occ = forward_backward_occlusion(flow, bwd, self.occlusion_thresh)
				self._last_occlusion = occ
			elif not self._warned_occ:
				print("Warning: occlusion-aware smoothing is not supported for this estimator "
				      "(non-standard flow convention); ignoring --motion-occlusion.")
				self._warned_occ = True

		# causal spatiotemporal window over recent flows
		if self.temporal_window > 1:
			self._hist.append(flow.copy())
			self._hist = self._hist[-self.temporal_window:]
			if len(self._hist) > 1:
				weights = _time_weights(len(self._hist))
				flow = np.average(np.stack(self._hist, axis=0), axis=0, weights=weights).astype(np.float32)

		flow = smooth_temporal(flow, prev_flow, alpha, occ)
		return flow

	def transfer_motion(self, flow, destframe, reverse_order=False):
		''' motion-compensate destframe by the (inverse) flow. standard convention. '''
		arr = np.asarray(destframe)
		h, w = arr.shape[:2]
		xx, yy = np.meshgrid(np.arange(w), np.arange(h))
		map_x = np.clip(xx - flow[..., 0], 0, w - 1).astype(np.float32)
		map_y = np.clip(yy - flow[..., 1], 0, h - 1).astype(np.float32)
		warped = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR)
		return PIL.Image.fromarray(warped)

	def flow_to_image(self, flow):
		''' direction -> hue, magnitude -> brightness '''
		hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
		hsv[..., 1] = 255
		mag, ang = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))
		hsv[..., 0] = ang * 180 / np.pi / 2
		hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		return PIL.Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))


# ---------------------------------------------------------------------------------------------
# RAFT (torchvision) - raw inference, transfer and flow_to_image kept byte-for-byte from the
# original implementation so existing GPU results are unchanged. Only the smoothing now lives
# in the base class; with the default alpha=0/sigma=0 it is a no-op, so behaviour is identical.
# ---------------------------------------------------------------------------------------------

class RAFTMotionEstimator(MotionEstimator):
	standard_convention = False  # RAFT's flow axes are swapped; occlusion warp is disabled

	def __init__(self, variant='large', **kwargs):
		super().__init__(**kwargs)
		if variant == 'small':
			self.raft_weights = Raft_Small_Weights.DEFAULT
			model_fn = raft_small
			weights = Raft_Small_Weights.DEFAULT
		else:
			self.raft_weights = Raft_Large_Weights.DEFAULT
			model_fn = raft_large
			weights = Raft_Large_Weights.DEFAULT
		self.raft_transforms = self.raft_weights.transforms()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.raft_model = model_fn(weights=weights, progress=False).to(self.device)
		self.raft_model = self.raft_model.eval()

	def _estimate_raw(self, srcframe1, srcframe2):
		# (verbatim from the original RAFTMotionEstimator.estimate_motion, smoothing removed)
		img1_batch = torch.tensor(np.asarray(srcframe1).transpose((2, 1, 0))).unsqueeze(0)
		img2_batch = torch.tensor(np.asarray(srcframe2).transpose((2, 1, 0))).unsqueeze(0)
		img1_batch = torchvision.transforms.functional.resize(img1_batch, size=[(srcframe1.width // 8) * 8, (srcframe1.height // 8) * 8], antialias=False)
		img2_batch = torchvision.transforms.functional.resize(img2_batch, size=[(srcframe2.width // 8) * 8, (srcframe2.height // 8) * 8], antialias=False)
		img1_batch, img2_batch = self.raft_transforms(img1_batch, img2_batch, )
		list_of_flows = self.raft_model(img1_batch.to(self.device), img2_batch.to(self.device))
		flow = list_of_flows[0].cpu().detach().numpy()
		if len(flow.shape) == 4:
			if flow.shape[0] == 1:
				flow = flow[0]
		if flow.shape[0] == 2:
			flow = np.transpose(flow, (2, 1, 0))
		return flow

	def transfer_motion(self, flow, destframe, reverse_order=True):
		# (verbatim from the original) reverse_order=True because RAFT's y axis is the first plane
		if len(flow.shape) == 4:
			flow = np.transpose(flow[0], (2, 1, 0))
		inv_flow = -flow
		destframe = np.asarray(destframe)
		h, w = destframe.shape[:2]
		x, y = np.meshgrid(np.arange(w), np.arange(h))
		x_inv = np.round(x + inv_flow[..., 0 if not reverse_order else 1]).astype(np.float32)
		y_inv = np.round(y + inv_flow[..., 1 if not reverse_order else 0]).astype(np.float32)
		x_inv = np.clip(x_inv, 0, w - 1)
		y_inv = np.clip(y_inv, 0, h - 1)
		warped = cv2.remap(np.asarray(destframe), x_inv, y_inv, cv2.INTER_LINEAR)
		return PIL.Image.fromarray(warped)

	def flow_to_image(self, flow):
		# (verbatim from the original)
		if len(flow.shape) == 4:
			flow = np.transpose(flow[0], (2, 1, 0))
		hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
		hsv[..., 1] = 255
		mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
		hsv[..., 0] = ang * 180 / np.pi / 2
		hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
		return PIL.Image.fromarray(rgb)


# ---------------------------------------------------------------------------------------------
# cv2 estimators (CPU) - standard flow convention, use the shared base transfer/flow_to_image
# ---------------------------------------------------------------------------------------------

def _gray(pil):
	return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2GRAY)


class FarnebackMotionEstimator(MotionEstimator):
	def _estimate_raw(self, prev_pil, cur_pil):
		g1, g2 = _gray(prev_pil), _gray(cur_pil)
		flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		return flow.astype(np.float32)


class DISMotionEstimator(MotionEstimator):
	def __init__(self, preset='medium', **kwargs):
		super().__init__(**kwargs)
		presets = {
			'ultrafast': cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
			'fast': cv2.DISOPTICAL_FLOW_PRESET_FAST,
			'medium': cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
		}
		self.dis = cv2.DISOpticalFlow_create(presets.get(preset, cv2.DISOPTICAL_FLOW_PRESET_MEDIUM))

	def _estimate_raw(self, prev_pil, cur_pil):
		g1, g2 = _gray(prev_pil), _gray(cur_pil)
		flow = self.dis.calc(g1, g2, None)
		return flow.astype(np.float32)


# ---------------------------------------------------------------------------------------------

ESTIMATORS = {
	'raft': ('large', RAFTMotionEstimator),
	'raft_large': ('large', RAFTMotionEstimator),
	'raft_small': ('small', RAFTMotionEstimator),
	'farneback': (None, FarnebackMotionEstimator),
	'dis': (None, DISMotionEstimator),
}


def create_estimator(name='raft', **kwargs):
	''' build a motion estimator by name, passing smoothing options as kwargs. '''
	name = (name or 'raft').lower()
	if name not in ESTIMATORS:
		raise ValueError(f"unknown motion estimator {name!r}; choices: {', '.join(sorted(ESTIMATORS))}")
	variant, cls = ESTIMATORS[name]
	if cls is RAFTMotionEstimator:
		return cls(variant=variant, **kwargs)
	return cls(**kwargs)
