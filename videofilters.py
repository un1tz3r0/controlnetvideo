"""
---------------------------------------------------------------------------------------------
videofilters.py

A self-contained, animatable video-filter library + standalone CLI for controlnetvideo.

Filters operate on RGB uint8 numpy frames and are composed into chains parsed from a
small spec language (audio-animate style):

    "<filter> <key>=<value>, <key>=<value>; <filter> ...; ..."

  - filters in a chain are separated by ';'
  - within a filter, params are separated by ','
  - a value is either a constant (number / #hexcolor / name) or an animation curve
    using the same 'L x y' / 'C x y a b c d' piecewise grammar as extract_beats.

Each filter param of numeric type may be animated. Curves are evaluated at a position
x (0..1) supplied per-frame: in the standalone CLI x is the clip position (or the
fractional bar position when --audio-from is given), and when imported into
controlnetvideo.py x is whatever animation position that pipeline passes in.

Run `python videofilters.py --list-filters` to see every filter and its params.

    by Victor Condino <un1tz3r0@gmail.com> - 2026
---------------------------------------------------------------------------------------------
"""

import math
import numpy as np
import cv2
import PIL.Image
import PIL.ImageColor

# ---------------------------------------------------------------------------------------------
# animation curve support
#
# we reuse extract_beats' piecewise-curve parser when it is importable (so the curve grammar
# stays identical to the rest of the project), but fall back to a small local implementation
# so this module can be imported and run standalone even when madmom/pydub are not installed.
# ---------------------------------------------------------------------------------------------

try:
	from extract_beats import parse_piecewise_function as _parse_curve  # type: ignore
except Exception:
	_parse_curve = None


def _parse_float_or_fraction(s, whole=1.0):
	s = s.strip()
	if '/' in s:
		num, denom = s.split('/')
		return (float(num) / float(denom)) * whole
	return float(s)


class _LocalPiecewise:
	"""minimal piecewise linear/cubic-bezier curve, grammar-compatible with extract_beats."""

	def __init__(self, knots):
		# knots: sorted list of (x, y, kind, params)
		self._knots = knots

	@staticmethod
	def _ease(t, x1, y1, x2, y2):
		# cubic-bezier easing evaluated by sampling t directly (good enough for animation)
		cx = 3.0 * x1
		bx = 3.0 * (x2 - x1) - cx
		ax = 1.0 - cx - bx
		cy = 3.0 * y1
		by = 3.0 * (y2 - y1) - cy
		ay = 1.0 - cy - by
		return ((ay * t + by) * t + cy) * t

	def __call__(self, x):
		ks = self._knots
		if not ks:
			return 0.0
		if x <= ks[0][0]:
			return ks[0][1]
		if x >= ks[-1][0]:
			return ks[-1][1]
		for i in range(1, len(ks)):
			x0, y0, _, _ = ks[i - 1]
			x1, y1, kind, params = ks[i]
			if x <= x1:
				if x1 == x0:
					return y1
				t = (x - x0) / (x1 - x0)
				if kind == 'C' and params is not None:
					return y0 + (y1 - y0) * self._ease(t, params[0], params[1], params[2], params[3])
				return y0 + (y1 - y0) * t
		return ks[-1][1]


def _local_parse_curve(s):
	words = s.strip().split()
	commands = []
	cur = []
	for w in words:
		if w.isalpha():
			if cur:
				commands.append(cur)
			cur = [w]
		else:
			cur.append(w)
	if cur:
		commands.append(cur)

	knots = [(0.0, 0.0, 'L', None)]
	for c in commands:
		op = c[0].upper()
		if op == 'L':
			x = _parse_float_or_fraction(c[1]); y = _parse_float_or_fraction(c[2])
			knots.append((x, y, 'L', None))
		elif op == 'C':
			x = _parse_float_or_fraction(c[1]); y = _parse_float_or_fraction(c[2])
			cx1 = _parse_float_or_fraction(c[3]); cy1 = _parse_float_or_fraction(c[4])
			cx2 = _parse_float_or_fraction(c[5]); cy2 = _parse_float_or_fraction(c[6])
			knots.append((x, y, 'C', (cx1, cy1, cx2, cy2)))
		else:
			raise ValueError(f"invalid curve segment: {' '.join(c)!r}")
	# drop the implicit (0,0) knot if the first real knot is also at x<=0
	if len(knots) > 1 and knots[1][0] <= 0.0:
		knots = knots[1:]
	knots.sort(key=lambda k: k[0])
	return _LocalPiecewise(knots)


def parse_curve(s):
	"""parse a piecewise animation curve string into a callable f(x)->y."""
	if _parse_curve is not None:
		return _parse_curve(s)
	return _local_parse_curve(s)


# ---------------------------------------------------------------------------------------------
# image / color helpers
# ---------------------------------------------------------------------------------------------

def to_rgb_u8(image):
	"""coerce PIL image / ndarray to an HxWx3 uint8 RGB ndarray."""
	if isinstance(image, PIL.Image.Image):
		return np.asarray(image.convert("RGB"))
	arr = np.asarray(image)
	if arr.dtype != np.uint8:
		arr = np.clip(arr, 0, 255).astype(np.uint8)
	if arr.ndim == 2:
		arr = np.stack([arr] * 3, axis=-1)
	if arr.shape[-1] == 4:
		arr = arr[..., :3]
	return np.ascontiguousarray(arr)


def parse_color(s):
	"""parse '#rgb', '#rrggbb', or a CSS color name into an (r,g,b) uint8 tuple."""
	if isinstance(s, (tuple, list)) and len(s) >= 3:
		return tuple(int(v) for v in s[:3])
	return tuple(PIL.ImageColor.getrgb(str(s).strip()))


def _f32(arr):
	return arr.astype(np.float32) / 255.0


def _u8(arr):
	return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


# blend modes operate on float arrays in 0..1
BLEND_MODES = {
	'normal':     lambda b, o: o,
	'multiply':   lambda b, o: b * o,
	'screen':     lambda b, o: 1.0 - (1.0 - b) * (1.0 - o),
	'overlay':    lambda b, o: np.where(b < 0.5, 2 * b * o, 1.0 - 2 * (1.0 - b) * (1.0 - o)),
	'softlight':  lambda b, o: np.where(o < 0.5, b - (1 - 2 * o) * b * (1 - b),
	                                     b + (2 * o - 1) * (np.where(b < 0.25, ((16 * b - 12) * b + 4) * b, np.sqrt(np.clip(b, 0, 1))) - b)),
	'add':        lambda b, o: np.clip(b + o, 0.0, 1.0),
	'subtract':   lambda b, o: np.clip(b - o, 0.0, 1.0),
	'difference': lambda b, o: np.abs(b - o),
	'lighten':    lambda b, o: np.maximum(b, o),
	'darken':     lambda b, o: np.minimum(b, o),
}


def blend_over(base_u8, over_u8, mode='normal', amount=1.0, mask=None):
	"""blend over_u8 onto base_u8 using the named mode; mask is an optional HxW float 0..1."""
	b = _f32(base_u8)
	o = _f32(over_u8)
	fn = BLEND_MODES.get(mode, BLEND_MODES['normal'])
	res = fn(b, o)
	a = float(amount)
	if mask is not None:
		a = a * mask[..., None]
	out = b * (1.0 - a) + res * a
	return _u8(out)


def smoothstep(edge0, edge1, x):
	if edge1 == edge0:
		return (x >= edge1).astype(np.float32)
	t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
	return t * t * (3.0 - 2.0 * t)


_BORDER_MODES = {
	'wrap': cv2.BORDER_WRAP,
	'reflect': cv2.BORDER_REFLECT_101,
	'clamp': cv2.BORDER_REPLICATE,
	'black': cv2.BORDER_CONSTANT,
}


# ---------------------------------------------------------------------------------------------
# filter registry
#
# each filter is a function fn(frame_u8, params, state, ctx) -> frame_u8
#   - params: dict of already-evaluated param values for this frame
#   - state:  per-filter-instance dict, persists across frames (for temporal filters)
#   - ctx:    FrameContext (frame index, time, fps, curve position x, depth provider)
# schema maps param name -> (type, default), type in float/int/color/str/bool.
# ---------------------------------------------------------------------------------------------

class FilterDef:
	def __init__(self, name, fn, schema, needs_depth=False, doc=""):
		self.name = name
		self.fn = fn
		self.schema = schema
		self.needs_depth = needs_depth
		self.doc = doc


FILTERS = {}


def register(name, schema, needs_depth=False, doc=""):
	def deco(fn):
		FILTERS[name] = FilterDef(name, fn, schema, needs_depth, doc or (fn.__doc__ or "").strip())
		return fn
	return deco


class FrameContext:
	def __init__(self, n, t, fps, x, depth_provider=None):
		self.n = n
		self.t = t
		self.fps = fps
		self.dt = 1.0 / fps if fps else 0.0
		self.x = x
		self.depth_provider = depth_provider
		self._depth_cache = {}

	def get_depth(self, frame_u8):
		"""normalized depth (HxW float 0..1, higher = nearer/foreground) for the given frame."""
		if self.depth_provider is None:
			self.depth_provider = DPTDepthProvider()
		key = id(frame_u8)
		if key not in self._depth_cache:
			self._depth_cache[key] = self.depth_provider(frame_u8)
		return self._depth_cache[key]


# ----- blur / denoise / sharpen --------------------------------------------------------------

@register('blur', {'sigma': ('float', 2.0)}, doc="gaussian blur")
def _blur(frame, p, state, ctx):
	s = max(0.0, p['sigma'])
	if s <= 0:
		return frame
	return cv2.GaussianBlur(frame, (0, 0), s)


@register('boxblur', {'size': ('int', 3)}, doc="box / average blur")
def _boxblur(frame, p, state, ctx):
	k = max(1, int(p['size']))
	return cv2.blur(frame, (k, k))


@register('bilateral', {'d': ('int', 9), 'color': ('float', 50.0), 'space': ('float', 50.0)},
          doc="edge-preserving bilateral filter")
def _bilateral(frame, p, state, ctx):
	return cv2.bilateralFilter(frame, int(p['d']), p['color'], p['space'])


@register('edgepreserve', {'sigma_s': ('float', 60.0), 'sigma_r': ('float', 0.4), 'recursive': ('bool', True)},
          doc="cv2 edge-preserving smoothing filter")
def _edgepreserve(frame, p, state, ctx):
	flags = cv2.RECURS_FILTER if p['recursive'] else cv2.NORMCONV_FILTER
	return cv2.edgePreservingFilter(frame, flags=flags, sigma_s=p['sigma_s'], sigma_r=p['sigma_r'])


@register('nlmeans', {'h': ('float', 6.0), 'template': ('int', 7), 'search': ('int', 21)},
          doc="non-local means denoising (slow)")
def _nlmeans(frame, p, state, ctx):
	return cv2.fastNlMeansDenoisingColored(frame, None, p['h'], p['h'], int(p['template']), int(p['search']))


@register('tdenoise', {'alpha': ('float', 0.5)},
          doc="temporal exponential denoise (blend with previous frame)")
def _tdenoise(frame, p, state, ctx):
	a = float(np.clip(p['alpha'], 0.0, 1.0))
	prev = state.get('prev')
	if prev is None or prev.shape != frame.shape:
		out = frame
	else:
		out = (prev.astype(np.float32) * a + frame.astype(np.float32) * (1.0 - a)).astype(np.uint8)
	state['prev'] = out
	return out


@register('sharpen', {'amount': ('float', 1.0), 'sigma': ('float', 2.0)},
          doc="unsharp-mask sharpening")
def _sharpen(frame, p, state, ctx):
	if p['amount'] <= 0:
		return frame
	blurred = cv2.GaussianBlur(frame, (0, 0), max(0.1, p['sigma']))
	f = frame.astype(np.float32)
	out = f + p['amount'] * (f - blurred.astype(np.float32))
	return np.clip(out, 0, 255).astype(np.uint8)


# ----- color modifications -------------------------------------------------------------------

def _to_hls(frame_u8):
	return cv2.cvtColor(_f32(frame_u8), cv2.COLOR_RGB2HLS)  # H 0..360, L/S 0..1


def _from_hls(hls):
	return _u8(cv2.cvtColor(hls, cv2.COLOR_HLS2RGB))


@register('hsl', {'hue': ('float', 0.0), 'sat': ('float', 1.0), 'light': ('float', 1.0)},
          doc="HSL adjust: hue shift (deg), saturation mult, lightness mult")
def _hsl(frame, p, state, ctx):
	hls = _to_hls(frame)
	hls[..., 0] = np.mod(hls[..., 0] + p['hue'], 360.0)
	hls[..., 1] = np.clip(hls[..., 1] * p['light'], 0.0, 1.0)
	hls[..., 2] = np.clip(hls[..., 2] * p['sat'], 0.0, 1.0)
	return _from_hls(hls)


@register('hueshift', {'deg': ('float', 0.0)}, doc="rotate hue by degrees")
def _hueshift(frame, p, state, ctx):
	hls = _to_hls(frame)
	hls[..., 0] = np.mod(hls[..., 0] + p['deg'], 360.0)
	return _from_hls(hls)


@register('saturation', {'amount': ('float', 1.0)}, doc="multiply saturation")
def _saturation(frame, p, state, ctx):
	hls = _to_hls(frame)
	hls[..., 2] = np.clip(hls[..., 2] * p['amount'], 0.0, 1.0)
	return _from_hls(hls)


@register('brightness', {'amount': ('float', 0.0)}, doc="add brightness (-1..1)")
def _brightness(frame, p, state, ctx):
	out = _f32(frame) + p['amount']
	return _u8(np.clip(out, 0.0, 1.0))


@register('contrast', {'amount': ('float', 1.0)}, doc="scale contrast about mid-gray")
def _contrast(frame, p, state, ctx):
	out = (_f32(frame) - 0.5) * p['amount'] + 0.5
	return _u8(np.clip(out, 0.0, 1.0))


@register('gamma', {'value': ('float', 1.0)}, doc="gamma correction")
def _gamma(frame, p, state, ctx):
	g = max(1e-3, p['value'])
	return _u8(np.power(_f32(frame), 1.0 / g))


# ----- geometric distortions (origin + direction + velocity, accumulated over time) ----------

def _warp(frame, M, border):
	h, w = frame.shape[:2]
	return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
	                      borderMode=_BORDER_MODES.get(border, cv2.BORDER_REFLECT_101))


@register('pan', {'vx': ('float', 0.0), 'vy': ('float', 0.0), 'edge': ('str', 'reflect')},
          doc="pan/translate; vx,vy in pixels per second (accumulated)")
def _pan(frame, p, state, ctx):
	tx = state.get('tx', 0.0) + p['vx'] * ctx.dt
	ty = state.get('ty', 0.0) + p['vy'] * ctx.dt
	state['tx'], state['ty'] = tx, ty
	M = np.float32([[1, 0, tx], [0, 1, ty]])
	return _warp(frame, M, p['edge'])


@register('zoom', {'rate': ('float', 0.0), 'ox': ('float', 0.5), 'oy': ('float', 0.5), 'edge': ('str', 'reflect')},
          doc="zoom about origin (ox,oy fraction); rate = fractional zoom per second")
def _zoom(frame, p, state, ctx):
	scale = state.get('scale', 1.0) * (1.0 + p['rate'] * ctx.dt)
	state['scale'] = scale
	h, w = frame.shape[:2]
	cx, cy = p['ox'] * w, p['oy'] * h
	M = cv2.getRotationMatrix2D((cx, cy), 0.0, scale)
	return _warp(frame, M, p['edge'])


@register('spin', {'speed': ('float', 0.0), 'ox': ('float', 0.5), 'oy': ('float', 0.5), 'edge': ('str', 'reflect')},
          doc="rotate about origin; speed = degrees per second")
def _spin(frame, p, state, ctx):
	angle = state.get('angle', 0.0) + p['speed'] * ctx.dt
	state['angle'] = angle
	h, w = frame.shape[:2]
	cx, cy = p['ox'] * w, p['oy'] * h
	M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
	return _warp(frame, M, p['edge'])


# ----- detector / depth-region effects -------------------------------------------------------

@register('lineart', {'low': ('float', 80.0), 'high': ('float', 160.0),
                      'color': ('color', '#000000'), 'amount': ('float', 1.0),
                      'blend': ('str', 'multiply'), 'thickness': ('int', 1)},
          doc="overlay canny lineart edges over the frame")
def _lineart(frame, p, state, ctx):
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	edges = cv2.Canny(gray, p['low'], p['high'])
	if int(p['thickness']) > 1:
		edges = cv2.dilate(edges, np.ones((int(p['thickness']),) * 2, np.uint8))
	mask = (edges.astype(np.float32) / 255.0)
	over = np.zeros_like(frame)
	over[:] = parse_color(p['color'])
	return blend_over(frame, over, p['blend'], p['amount'], mask)


def _depth_mask(frame, ctx, threshold, feather, region):
	depth = ctx.get_depth(frame)
	if depth.shape != frame.shape[:2]:
		depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
	fg = smoothstep(threshold - feather, threshold + feather, depth)  # 1 = foreground
	return fg if region == 'fg' else (1.0 - fg)


@register('depthfx', {'effect': ('str', 'blur'), 'region': ('str', 'bg'),
                      'threshold': ('float', 0.5), 'feather': ('float', 0.05),
                      'amount': ('float', 1.0), 'blur': ('float', 8.0),
                      'color': ('color', '#000000'), 'offset': ('int', 6), 'width': ('int', 3)},
          needs_depth=True,
          doc="depth-thresholded region effect: effect=brighten|darken|blur|outline|"
              "innershadow|outershadow|innerblur|outerblur, region=fg|bg")
def _depthfx(frame, p, state, ctx):
	effect = p['effect']
	mask = _depth_mask(frame, ctx, p['threshold'], p['feather'], p['region'])
	m = mask[..., None]
	f = frame.astype(np.float32)

	if effect == 'brighten':
		out = f + (255.0 * p['amount']) * m
	elif effect == 'darken':
		out = f - (255.0 * p['amount']) * m
	elif effect in ('blur', 'innerblur', 'outerblur'):
		blurred = cv2.GaussianBlur(frame, (0, 0), max(0.1, p['blur'])).astype(np.float32)
		# 'blur' blurs the selected region; inner/outer name the same thing explicitly
		out = f * (1.0 - m) + blurred * m
	elif effect == 'outline':
		hard = (mask > 0.5).astype(np.uint8) * 255
		grad = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT,
		                        np.ones((max(1, int(p['width'])),) * 2, np.uint8))
		em = (grad.astype(np.float32) / 255.0)[..., None]
		col = np.array(parse_color(p['color']), np.float32)
		out = f * (1.0 - em * p['amount']) + col * (em * p['amount'])
	elif effect in ('innershadow', 'outershadow'):
		off = int(p['offset'])
		hard = (mask > 0.5).astype(np.float32)
		shifted = np.roll(np.roll(hard, off, axis=0), off, axis=1)
		shifted = cv2.GaussianBlur(shifted, (0, 0), max(0.1, p['blur']))
		if effect == 'innershadow':
			shadow = np.clip(hard - shifted, 0.0, 1.0) * hard
		else:
			shadow = np.clip(shifted - hard, 0.0, 1.0) * (1.0 - hard)
		sm = (shadow * p['amount'])[..., None]
		col = np.array(parse_color(p['color']), np.float32)
		out = f * (1.0 - sm) + col * sm
	else:
		raise ValueError(f"unknown depthfx effect: {effect!r}")

	return np.clip(out, 0, 255).astype(np.uint8)


# ----- animatable pattern overlays -----------------------------------------------------------

def _pattern_coords(shape, angle_deg, scale, phase):
	h, w = shape
	yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
	a = math.radians(angle_deg)
	ca, sa = math.cos(a), math.sin(a)
	u = (xx * ca + yy * sa) / max(1e-3, scale)
	v = (-xx * sa + yy * ca) / max(1e-3, scale)
	return u + phase, v


def _make_pattern(ptype, shape, p, t):
	scale = max(1.0, p['scale'])
	phase = p['speed'] * t
	u, v = _pattern_coords(shape, p['angle'], scale, phase)
	duty = float(np.clip(p['duty'], 0.0, 1.0))

	if ptype == 'stripes':
		m = (np.mod(u, 1.0) < duty)
	elif ptype == 'dashed':
		m = (np.mod(u, 1.0) < duty) & (np.mod(v, 1.0) < float(np.clip(p['dash'], 0.0, 1.0)))
	elif ptype == 'zigzag':
		tri = np.abs(np.mod(v, 1.0) - 0.5) * 2.0
		m = (np.mod(u + tri, 1.0) < duty)
	elif ptype == 'chevron':
		tri = np.abs(np.mod(v, 1.0) - 0.5) * 2.0
		m = (np.mod(u + tri * 0.5, 1.0) < duty) | (np.mod(u - tri * 0.5, 1.0) < duty)
	elif ptype == 'checker':
		m = (np.mod(np.floor(u) + np.floor(v), 2.0) < 1.0)
	elif ptype == 'grid':
		lw = float(np.clip(p['duty'], 0.02, 0.5))
		m = (np.mod(u, 1.0) < lw) | (np.mod(v, 1.0) < lw)
	elif ptype == 'polkadot':
		du = np.mod(u, 1.0) - 0.5
		dv = np.mod(v, 1.0) - 0.5
		m = (du * du + dv * dv) < (0.5 * duty) ** 2
	elif ptype == 'honeycomb':
		# hex grid lines from three stripe directions 60deg apart
		acc = np.zeros(shape, np.float32)
		lw = float(np.clip(p['duty'], 0.02, 0.4))
		for da in (0.0, 60.0, 120.0):
			uu, _ = _pattern_coords(shape, p['angle'] + da, scale, phase)
			acc = np.maximum(acc, (np.mod(uu, 1.0) < lw).astype(np.float32))
		m = acc > 0.5
	else:
		raise ValueError(f"unknown pattern type: {ptype!r}")

	out = np.empty((shape[0], shape[1], 3), np.uint8)
	out[m] = parse_color(p['color1'])
	out[~m] = parse_color(p['color2'])
	return out


_PATTERN_SCHEMA = {
	'type': ('str', 'stripes'),
	'scale': ('float', 24.0),
	'angle': ('float', 45.0),
	'speed': ('float', 0.0),
	'duty': ('float', 0.5),
	'dash': ('float', 0.5),
	'color1': ('color', '#ffffff'),
	'color2': ('color', '#000000'),
	'blend': ('str', 'normal'),
	'amount': ('float', 0.5),
}


@register('pattern', _PATTERN_SCHEMA,
          doc="overlay an animatable pattern: type=stripes|dashed|zigzag|chevron|checker|"
              "grid|polkadot|honeycomb; params scale,angle,speed,duty,dash,color1,color2,blend,amount")
def _pattern(frame, p, state, ctx):
	over = _make_pattern(p['type'], frame.shape[:2], p, ctx.t)
	return blend_over(frame, over, p['blend'], p['amount'])


# ---------------------------------------------------------------------------------------------
# default depth provider (lazy; only loaded if a depth filter is actually used)
# ---------------------------------------------------------------------------------------------

class DPTDepthProvider:
	"""lazy DPT/MiDaS depth estimator returning normalized depth (1 = nearest)."""

	def __init__(self, model_name="Intel/dpt-hybrid-midas", device=None):
		self.model_name = model_name
		self.device = device
		self._model = None
		self._proc = None

	def _load(self):
		try:
			import torch
			from transformers import DPTImageProcessor, DPTForDepthEstimation
		except Exception as e:
			raise RuntimeError(
				"depth filters need torch + transformers installed, or pass a depth_provider. "
				f"(import failed: {e})")
		self._torch = torch
		self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
		self._model = DPTForDepthEstimation.from_pretrained(self.model_name).to(self.device).eval()
		self._proc = DPTImageProcessor.from_pretrained(self.model_name)

	def __call__(self, frame_u8):
		if self._model is None:
			self._load()
		torch = self._torch
		inputs = self._proc(images=PIL.Image.fromarray(frame_u8), return_tensors="pt").pixel_values.to(self.device)
		with torch.no_grad():
			depth = self._model(inputs).predicted_depth
		d = depth.squeeze().detach().cpu().numpy().astype(np.float32)
		d = cv2.resize(d, (frame_u8.shape[1], frame_u8.shape[0]))
		dmin, dmax = float(d.min()), float(d.max())
		if dmax > dmin:
			d = (d - dmin) / (dmax - dmin)
		return d


# ---------------------------------------------------------------------------------------------
# spec parsing + chain
# ---------------------------------------------------------------------------------------------

class ParamValue:
	__slots__ = ('typ', 'const', 'curve')

	def __init__(self, typ, const=None, curve=None):
		self.typ = typ
		self.const = const
		self.curve = curve

	def eval(self, x):
		if self.curve is not None:
			v = float(self.curve(x))
			return int(round(v)) if self.typ == 'int' else v
		return self.const


def _make_param(typ, raw):
	raw = raw.strip()
	if typ in ('float', 'int'):
		toks = raw.split()
		if toks and toks[0][:1].isalpha():  # looks like an animation curve (L.../C...)
			return ParamValue(typ, curve=parse_curve(raw))
		v = float(raw)
		return ParamValue(typ, const=(int(round(v)) if typ == 'int' else v))
	if typ == 'color':
		return ParamValue(typ, const=parse_color(raw))
	if typ == 'bool':
		return ParamValue(typ, const=raw.lower() in ('1', 'true', 'yes', 'on'))
	return ParamValue(typ, const=raw)  # str


class FilterChain:
	def __init__(self, steps, fps=30.0, depth_provider=None):
		# steps: list of (FilterDef, {name: ParamValue})
		self.steps = steps
		self.fps = fps
		self.depth_provider = depth_provider
		self.requires_depth = any(fd.needs_depth for fd, _ in steps)
		self._states = [{} for _ in steps]
		self._n = 0

	def reset(self):
		self._states = [{} for _ in self.steps]
		self._n = 0

	def __call__(self, image, x=0.0, n=None, t=None, fps=None, depth_provider=None):
		fps = fps or self.fps
		n = self._n if n is None else n
		t = (n / fps) if (t is None and fps) else (t or 0.0)
		dp = depth_provider or self.depth_provider
		if self.requires_depth and dp is None:
			dp = self.depth_provider = DPTDepthProvider()
		ctx = FrameContext(n, t, fps, x, dp)

		was_pil = isinstance(image, PIL.Image.Image)
		arr = to_rgb_u8(image)
		for (fd, params), state in zip(self.steps, self._states):
			ev = {}
			for pname, (ptype, default) in fd.schema.items():
				pv = params.get(pname)
				ev[pname] = pv.eval(x) if pv is not None else default
			arr = fd.fn(arr, ev, state, ctx)
		self._n = n + 1
		return PIL.Image.fromarray(arr) if was_pil else arr


def parse_video_filter_spec(spec, fps=30.0, depth_provider=None):
	"""parse a filter-chain spec string into a callable FilterChain.

	grammar:  "<name> k=v, k=v; <name> k=v; ..."  (';' between filters, ',' between params)
	values are constants or 'L x y' / 'C x y a b c d' animation curves.
	Returns a FilterChain; call chain(image, x) per frame.
	"""
	if spec is None or not str(spec).strip():
		return FilterChain([], fps=fps, depth_provider=depth_provider)

	steps = []
	for seg in spec.split(';'):
		seg = seg.strip()
		if not seg:
			continue
		# split filter name (first whitespace token) from its params
		if ' ' in seg:
			name, rest = seg.split(None, 1)
		else:
			name, rest = seg, ''
		name = name.strip()
		if name not in FILTERS:
			raise ValueError(f"unknown filter {name!r}. known: {', '.join(sorted(FILTERS))}")
		fd = FILTERS[name]
		params = {}
		for assign in rest.split(','):
			assign = assign.strip()
			if not assign:
				continue
			if '=' not in assign:
				raise ValueError(f"filter {name!r}: expected key=value, got {assign!r}")
			key, val = assign.split('=', 1)
			key = key.strip()
			if key not in fd.schema:
				raise ValueError(f"filter {name!r}: unknown param {key!r}. valid: {', '.join(fd.schema)}")
			params[key] = _make_param(fd.schema[key][0], val)
		steps.append((fd, params))

	return FilterChain(steps, fps=fps, depth_provider=depth_provider)


def list_filters():
	"""return a human-readable listing of all registered filters and their params."""
	lines = []
	for name in sorted(FILTERS):
		fd = FILTERS[name]
		params = ', '.join(f"{k}={d!r}" for k, (t, d) in fd.schema.items())
		tag = ' [needs depth]' if fd.needs_depth else ''
		lines.append(f"{name}{tag}\n    {fd.doc}\n    params: {params}")
	return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------
# standalone CLI
# ---------------------------------------------------------------------------------------------

def _process_video(input_video, output_video, chain, start_time, end_time, duration,
                   max_dimension, audio_from, position, output_bitrate, output_codec, fps_override):
	import moviepy as mp

	video = mp.VideoFileClip(input_video)

	# moviepy 2.x renamed subclip -> subclipped; support both
	def _subclip(clip, a, b=None):
		fn = getattr(clip, 'subclipped', None) or getattr(clip, 'subclip')
		return fn(a, b) if b is not None else fn(a)

	if end_time is not None:
		video = _subclip(video, 0, end_time)
	if start_time is not None:
		video = _subclip(video, start_time)
	if duration is not None:
		video = _subclip(video, 0, duration)

	w, h = video.size
	if max_dimension is not None and max(w, h) > max_dimension:
		if w >= h:
			w, h = max_dimension, int(round(h / w * max_dimension))
		else:
			w, h = int(round(w / h * max_dimension)), max_dimension

	fps = fps_override or video.fps
	chain.fps = fps

	beats = None
	if audio_from is not None:
		import extract_beats
		print(f"Analysing audio {audio_from!r} for tempo/beats...")
		_, _, beats = extract_beats.extract_beats(audio_from, fps=fps)

	total = max(1, int(round(video.duration * fps)))
	state = {'n': 0}

	def transform(get_frame, t):
		n = state['n']
		if beats is not None:
			bar, _ = extract_beats.time_to_songpos(t, beats)
			x = math.fmod(bar, 1.0)
		elif position == 'time':
			x = t
		else:
			x = (n / (total - 1)) if total > 1 else 0.0
		frame = PIL.Image.fromarray(get_frame(t)).resize((w, h))
		out = chain(frame, x=x, n=n, t=t, fps=fps)
		state['n'] = n + 1
		if n % 25 == 0:
			print(f"  frame {n}/{total}  t={t:.2f}s  x={x:.3f}")
		return np.asarray(out)

	out = video.transform(transform, keep_duration=True)
	print(f"Writing {output_video} ...")
	out.write_videofile(output_video, codec=output_codec, fps=fps, bitrate=output_bitrate)
	video.close()


def main():
	import click

	@click.command(context_settings=dict(help_option_names=['-h', '--help']))
	@click.argument('input_video', required=False, type=click.Path(exists=True))
	@click.argument('output_video', required=False, type=click.Path())
	@click.option('--filter', '-f', 'filter_spec', default=None,
	              help="filter-chain spec, e.g. 'blur sigma=2; hueshift deg=L 0 0 1 180'")
	@click.option('--list-filters', is_flag=True, default=False, help="list available filters and exit")
	@click.option('--start-time', type=float, default=None)
	@click.option('--end-time', type=float, default=None)
	@click.option('--duration', type=float, default=None)
	@click.option('--max-dimension', type=int, default=None, help="downscale so the longest side is at most this")
	@click.option('--audio-from', type=click.Path(exists=True), default=None,
	              help="drive curve position x from the fractional bar position of this audio file")
	@click.option('--position', type=click.Choice(['clip', 'time']), default='clip',
	              help="curve position basis when no --audio-from: 'clip' = 0..1 over clip, 'time' = seconds")
	@click.option('--fps', 'fps_override', type=float, default=None, help="override output fps")
	@click.option('--output-bitrate', type=str, default="16M")
	@click.option('--output-codec', type=str, default="libx264")
	def cli(input_video, output_video, filter_spec, list_filters,
	        start_time, end_time, duration, max_dimension, audio_from,
	        position, fps_override, output_bitrate, output_codec):
		"""Apply a chain of animatable video filters to INPUT_VIDEO, writing OUTPUT_VIDEO."""
		if list_filters:
			click.echo(list_filters_text())
			return
		if not input_video or not output_video:
			raise click.UsageError("INPUT_VIDEO and OUTPUT_VIDEO are required (or use --list-filters)")
		if not filter_spec:
			raise click.UsageError("a --filter chain spec is required")

		chain = parse_video_filter_spec(filter_spec, fps=fps_override or 30.0)
		print(f"Parsed {len(chain.steps)} filter(s); requires_depth={chain.requires_depth}")
		_process_video(input_video, output_video, chain, start_time, end_time, duration,
		               max_dimension, audio_from, position, output_bitrate, output_codec, fps_override)

	# bound here so the closure can see it without shadowing the flag name
	def list_filters_text():
		return list_filters()

	cli()


if __name__ == "__main__":
	main()
