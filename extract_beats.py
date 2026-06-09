from ast import Slice
import pathlib
import json
from pydub import AudioSegment
import madmom
from madmom.features import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
import moviepy as mp
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
import PIL
from tqdm import tqdm

def audio_to_signal(audio):
	'''
	conversion between audio library types, pydub -> madmom
	input: pydub audiosegment
	output: madmom signal object
	'''
	data = np.asarray(audio.get_array_of_samples())
	sig = madmom.audio.Signal(data=data, sample_rate=audio.frame_rate, num_channels=audio.channels)
	return sig

def audio_to_clip(audio):
	'''
	conversion between audio library types, pydub -> moviepy
	input: pydub AudioSegment
	output moviepy AudioArrayClip
	'''
	# workaround for pydub ignoring channel count in get_array_of_samples()... split to mono and then call, then
	# reassemble with numpy from mono channels
	data = np.asarray([np.asarray(ch.get_array_of_samples()) for ch in audio.split_to_mono()]).transpose((1,0))
	clip = AudioArrayClip(data, fps=audio.frame_rate)
	return clip

# ---------------------------------------------------------------------------------------------
# beat-extraction cache + per-beat/per-bar RMS intensity
#
# Analysing audio with madmom's RNN/DBN downbeat tracker is slow, so results are cached to a
# JSONL file next to the source (same name, '.jsonl' extension). Each line after the metadata
# header is one beat: its time (decimal seconds, and frame@fps if an fps is given), its beat
# number within the bar, and the RMS amplitude over the trailing beat- and bar-length windows
# (a measure of song intensity and buildup/drop). On a cache hit the slow analysis is skipped.
# ---------------------------------------------------------------------------------------------

CACHE_VERSION = 1


def beat_cache_path(filename, cache_path=None):
	''' default cache path for a source file: same name with a .jsonl extension '''
	if cache_path is not None:
		return pathlib.Path(cache_path)
	return pathlib.Path(filename).with_suffix('.jsonl')


def read_beat_cache(path):
	''' read a beat cache jsonl file, returning (meta_dict_or_None, list_of_entry_dicts) '''
	meta = None
	entries = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			if obj.get('_meta'):
				meta = obj
			else:
				entries.append(obj)
	return meta, entries


def write_beat_cache(path, entries, meta):
	''' write a beat cache jsonl file: one metadata header line followed by one line per beat '''
	with open(path, 'w') as f:
		f.write(json.dumps(meta) + "\n")
		for e in entries:
			f.write(json.dumps(e) + "\n")


def _cache_is_valid(meta, starttime, endtime, samplerate):
	''' a cache is reusable only if it covers the same crop and sample rate it was computed for '''
	if meta is None or meta.get('version') != CACHE_VERSION:
		return False
	def close(a, b):
		return abs((a if a is not None else 0.0) - (b if b is not None else 0.0)) < 1e-3
	return (close(meta.get('starttime'), starttime) and
	        close(meta.get('endtime'), endtime) and
	        int(meta.get('samplerate', -1)) == int(samplerate))


def _to_mono_float(sig):
	''' return (mono samples peak-normalized to -1..1, sample_rate) from a madmom Signal / ndarray.

	We normalize by the peak amplitude of the analysed region rather than by the dtype full
	scale, so the resulting rms is a 0..1 intensity measure regardless of bit depth or whether
	madmom handed us int16 samples or float samples still in integer range. '''
	data = np.asarray(sig)
	sr = int(getattr(sig, 'sample_rate', 44100))
	if data.ndim > 1:
		data = data.mean(axis=1)
	data = data.astype(np.float64)
	peak = float(np.max(np.abs(data))) if data.size else 0.0
	if peak > 0:
		data = data / peak
	return data, sr


def _rms_over(data, sr, t0, t1):
	''' rms amplitude (0..1 of full scale) of the mono samples in the time window [t0, t1] seconds '''
	if t1 <= t0:
		return 0.0
	i0 = max(0, int(round(t0 * sr)))
	i1 = min(len(data), int(round(t1 * sr)))
	if i1 <= i0:
		return 0.0
	seg = data[i0:i1]
	return float(np.sqrt(np.mean(seg * seg)))


def _build_beat_entries(raw_beats, sig, fps=None, show_progress=True):
	''' turn a madmom (time, beatnum) array into rich per-beat entry dicts with rms intensity '''
	data, sr = _to_mono_float(sig)
	beats_list = [(float(t), int(round(b))) for t, b in raw_beats]
	beats_per_bar = max((b for _, b in beats_list), default=4)
	entries = []
	it = tqdm(beats_list, desc="Computing beat/bar RMS", unit="beat") if show_progress else beats_list
	for i, (t, b) in enumerate(it):
		# beat window: from the previous beat up to this one
		beat_t0 = beats_list[i - 1][0] if i > 0 else 0.0
		beat_rms = _rms_over(data, sr, beat_t0, t)
		# bar window: from one bar (beats_per_bar beats) ago up to this one
		j = i - beats_per_bar
		bar_t0 = beats_list[j][0] if j >= 0 else 0.0
		bar_rms = _rms_over(data, sr, bar_t0, t)
		entry = {"time": round(t, 6), "beat": b,
		         "beat_rms": round(beat_rms, 6), "bar_rms": round(bar_rms, 6)}
		if fps:
			entry["frame"] = int(round(t * fps))
		entries.append(entry)
	return entries


def _run_downbeat_analysis(sig, est_seconds=None):
	''' run the (slow) madmom RNN+DBN downbeat tracker with a live progress bar '''
	import threading
	import time as _time

	print(f"Analysing audio for tempo and downbeats...")
	result = {}
	error = {}

	def _work():
		try:
			activations = RNNDownBeatProcessor()(sig)
			result['beats'] = DBNDownBeatTrackingProcessor(
				beats_per_bar=4, fps=100, min_bpm=55, max_bpm=185)(activations)
		except Exception as e:  # surfaced on the main thread below
			error['e'] = e

	worker = threading.Thread(target=_work, daemon=True)
	worker.start()

	# the analysis is one long blocking call with no native progress hook, so we show an
	# elapsed/ETA bar driven by a rough realtime estimate, snapping to 100% on completion
	total = max(1, int(round(est_seconds or 30)))
	start = _time.monotonic()
	with tqdm(total=total, desc="Downbeat RNN", unit="s",
	          bar_format="{desc}: {bar} {elapsed}<~{remaining}") as pbar:
		while worker.is_alive():
			worker.join(0.25)
			elapsed = _time.monotonic() - start
			pbar.n = min(int(elapsed), total - 1)
			pbar.refresh()
		pbar.n = total
		pbar.refresh()

	if 'e' in error:
		raise error['e']
	return result['beats']


def _get_beats_cached(sig, filename, starttime, endtime, fps, use_cache, cache_path, est_seconds=None):
	''' return (beats_list, entries), loading from / saving to the jsonl cache when possible '''
	cpath = beat_cache_path(filename, cache_path)
	samplerate = int(getattr(sig, 'sample_rate', 44100))

	if use_cache and cpath.exists():
		try:
			meta, entries = read_beat_cache(cpath)
			if _cache_is_valid(meta, starttime, endtime, samplerate):
				print(f"Reusing cached beat analysis from {cpath} ({len(entries)} beats)")
				# refresh the frame field for the requested fps without re-analysing
				if fps:
					for e in entries:
						e["frame"] = int(round(e["time"] * fps))
				beats = [(e["time"], e["beat"]) for e in entries]
				return beats, entries
			else:
				print(f"Ignoring stale beat cache {cpath} (crop/samplerate differs), re-analysing...")
		except Exception as e:
			print(f"Could not read beat cache {cpath} ({e}), re-analysing...")

	raw_beats = _run_downbeat_analysis(sig, est_seconds=est_seconds)
	entries = _build_beat_entries(raw_beats, sig, fps=fps)
	beats = [(e["time"], e["beat"]) for e in entries]

	if use_cache:
		meta = {"_meta": True, "version": CACHE_VERSION, "source": str(filename),
		        "starttime": starttime, "endtime": endtime, "samplerate": samplerate,
		        "beats_per_bar": 4, "fps": fps}
		try:
			write_beat_cache(cpath, entries, meta)
			print(f"Saved beat analysis cache to {cpath}")
		except Exception as e:
			print(f"Warning: could not write beat cache {cpath}: {e}")

	return beats, entries


def _print_beat_stats(beats):
	''' print a quick tempo/bar summary; robust to short clips with too few beats '''
	beatcounts = {n: 0 for n in [1, 2, 3, 4]}
	beatdiffs = {n: 0.0 for n in [1, 2, 3, 4]}
	lastbeat = {n: None for n in [1, 2, 3, 4]}
	for bt, bn in beats:
		bn = int(bn)
		if bn not in beatcounts:
			continue
		lt = lastbeat[bn]
		lastbeat[bn] = bt
		if lt is not None:
			beatdiffs[bn] += bt - lt
			beatcounts[bn] += 1
	avgs = [beatdiffs[b] / beatcounts[b] for b in [1, 2, 3, 4] if beatcounts[b] > 0]
	bars = sum(1 for bt, bn in beats if int(bn) == 1)
	if avgs:
		beatavg = sum(avgs) / len(avgs) / 4
		print(f"Done. Tempo: {60.0 / beatavg:.1f} BPM. Bars: {bars}, Beats: {len(beats)}")
	else:
		print(f"Done. {len(beats)} beats detected (too few for tempo stats).")


def _load_cropped_audio(filename, starttime, endtime, duration):
	''' load an audio file with pydub and crop it to the requested region; returns (audio, starttime, endtime) '''
	if starttime is None:
		starttime = 0.0
	if endtime is None and duration is not None:
		endtime = starttime + duration
	elif endtime is not None and duration is None:
		duration = endtime - starttime
	print(f"Reading audio track from file: {filename}")
	audio = AudioSegment.from_file(pathlib.Path(filename))
	origlen = len(audio) / 1000.0
	if endtime is None:
		endtime = len(audio) / 1000.0
	audio = audio[starttime * 1000:endtime * 1000]
	print(f"Read {origlen}s of audio samples, clipped from {starttime} to {endtime}")
	return audio, starttime, endtime


def extract_beats(filename, starttime=None, endtime=None, duration=None,
                  fps=None, use_cache=True, cache_path=None):
	'''
	analyse (or load from cache) the downbeats of an audio file.

	returns (audio, sig, beats) where beats is a list of (time_seconds, beat_number) pairs.
	Results are cached to a '<name>.jsonl' file next to the source; pass use_cache=False to
	force re-analysis or cache_path=... to override the location. If fps is given, the cached
	entries also carry a frame index. The richer per-beat data (rms intensity) is available
	via extract_beat_data().
	'''
	audio, starttime, endtime = _load_cropped_audio(filename, starttime, endtime, duration)
	sig = audio_to_signal(audio)
	beats, entries = _get_beats_cached(sig, filename, starttime, endtime, fps,
	                                   use_cache, cache_path, est_seconds=len(audio) / 1000.0)
	_print_beat_stats(beats)
	return audio, sig, beats


def extract_beat_data(filename, starttime=None, endtime=None, duration=None,
                      fps=None, use_cache=True, cache_path=None):
	'''
	like extract_beats(), but returns the rich list of per-beat entry dicts:
	{ "time": seconds, "beat": n, "beat_rms": ..., "bar_rms": ..., ["frame": k] }

	beat_rms / bar_rms are the rms amplitude over the trailing beat- and bar-length windows,
	a measure of intensity and buildup/drop useful for timing effect punch-in/out or edits.
	'''
	audio, starttime, endtime = _load_cropped_audio(filename, starttime, endtime, duration)
	sig = audio_to_signal(audio)
	beats, entries = _get_beats_cached(sig, filename, starttime, endtime, fps,
	                                   use_cache, cache_path, est_seconds=len(audio) / 1000.0)
	_print_beat_stats(beats)
	return entries

def count_bars(beats):
	count = 0
	for time, beat in beats:
		if int(beat) == 1:
			count = count + 1
	return count + 1

def slice_bars(audio, beats, start_bar, end_bar=None):
	if end_bar == None:
		end_bar = start_bar
	bar_ranges = []
	last_time = 0
	for time, beat in beats:
		if int(beat) == 1:
			bar_ranges.append((last_time, time))
			last_time = time
	start_time = bar_ranges[start_bar][0]
	end_time = bar_ranges[end_bar][1]
	return audio[start_time*1000:end_time*1000]

def slice_all_bars(audio, beats):
	for bar in range(0, count_bars(beats)):
		yield slice_bars(audio, beats, bar)

def songpos_to_time(t, beats):
	'''
	converts from fractional bar song position to seconds, according to the beats array returned by extract_beats()
	'''
	beats_per_bar = max([int(bn) for bt, bn in beats])
	beats_by_time = list(sorted(beats, key=lambda b: b[0]))
	bars_beats = []
	bar = 0
	beat = 0
	for bt, bn in beats_by_time:
		if int(bn) == 1:
			bar = bar + 1
		barpos = (int(bn) - 1) / beats_per_bar
		beat = beat + 1
		bars_beats.append((bt, bar + barpos, beat))
	prec_beat_bar = (0,0,0)
	succ_beat_bar = bars_beats[-1]
	for bt, barn, beatn in bars_beats:
		if barn <= t:
			prec_beat_bar = (bt, barn, beatn)
		if barn > t:
			succ_beat_bar = (bt, barn, beatn)
			break
	if succ_beat_bar == None:
		# past end
		return (prec_beat_bar[0], prec_beat_bar[0])
	pre_time, pre_bar, pre_beat = prec_beat_bar
	suc_time, suc_bar, suc_beat = succ_beat_bar
	pt = (t - pre_bar) / (suc_bar - pre_bar)
	t = (pre_time * (1-pt)) + (suc_time * pt)
	return t

def time_to_songpos(t, beats):
	'''
	gets the fractional bar and beat position of a time in seconds given the beat positions output by extract_beats
	'''
	beats_per_bar = max([int(bn) for bt, bn in beats])
	beats_by_time = list(sorted(beats, key=lambda b: b[0]))
	bars_beats = []
	bar = 0
	beat = 0
	for bt, bn in beats_by_time:
		if int(bn) == 1:
			bar = bar + 1
		barpos = (int(bn) - 1) / beats_per_bar
		beat = beat + 1
		bars_beats.append((bt, bar + barpos, beat))
	prec_beat_bar = (0,0,0)
	succ_beat_bar = bars_beats[-1]
	for bt, barn, beatn in bars_beats:
		if bt <= t:
			prec_beat_bar = (bt, barn, beatn)
		if bt > t:
			succ_beat_bar = (bt, barn, beatn)
			break
	if succ_beat_bar == None:
		# past end
		return (prec_beat_bar[1], prec_beat_bar[2])
	pre_time, pre_bar, pre_beat = prec_beat_bar
	suc_time, suc_bar, suc_beat = succ_beat_bar
	pt = (t - pre_time) / (suc_time - pre_time)
	bar = (pre_bar * (1-pt)) + (suc_bar * pt)
	beat = (pre_beat * (1-pt)) + (suc_beat * pt)
	return (bar, beat)

# ---------------------------------------------------------------
# cubic-bezier() easing functions, translated from javascript to
# python by bard.google.com
# ---------------------------------------------------------------

class CubicBezier:
	@staticmethod
	def A(a1, a2):
		"""
		Helper function for calculating the coefficient A in the cubic bezier formula.

		Args:
			a1: The first control point of the bezier curve.
			a2: The second control point of the bezier curve.

		Returns:
			The coefficient A in the cubic bezier formula.
		"""
		return 1.0 - 3.0 * a2 + 3.0 * a1

	@staticmethod
	def B(a1, a2):
		"""
		Helper function for calculating the coefficient B in the cubic bezier formula.

		Args:
			a1: The first control point of the bezier curve.
			a2: The second control point of the bezier curve.

		Returns:
			The coefficient B in the cubic bezier formula.
		"""
		return 3.0 * a2 - 6.0 * a1

	@staticmethod
	def C(a1):
		"""
		Helper function for calculating the coefficient C in the cubic bezier formula.

		Args:
			a1: The first control point of the bezier curve.

		Returns:
			The coefficient C in the cubic bezier formula.
		"""
		return 3.0 * a1

	@staticmethod
	def calc_bezier(t, a1, a2):
		"""
		Calculates the position of a point on a cubic bezier curve at a given time.

		Args:
			t: The normalized time (0.0 to 1.0) along the bezier curve.
			a1: The first control point of the bezier curve.
			a2: The second control point of the bezier curve.

		Returns:
			The position of the point on the bezier curve at time t.
		"""
		return ((A(a1, a2) * t + B(a1, a2)) * t + C(a1)) * t

	@staticmethod
	def get_slope(t, a1, a2):
		"""
		Calculates the slope of a cubic bezier curve at a given time.

		Args:
			t: The normalized time (0.0 to 1.0) along the bezier curve.
			a1: The first control point of the bezier curve.
			a2: The second control point of the bezier curve.

		Returns:
			The slope of the bezier curve at time t.
		"""
		return 3.0 * A(a1, a2) * t * t + 2.0 * B(a1, a2) * t + C(a1)

	def __init__(self, x1, y1, x2, y2):
		"""
		Initializes a cubic bezier curve.

		Args:
			x1: The x coordinate of the first control point.
			y1: The y coordinate of the first control point.
			x2: The x coordinate of the second control point.
			y2: The y coordinate of the second control point.
		"""
		self.cx = 3.0 * x1
		self.bx = 3.0 * (x2 - x1) - self.cx
		self.ax = 1.0 - self.cx - self.bx

		self.cy = 3.0 * y1
		self.by = 3.0 * (y2 - y1) - self.cy
		self.ay = 1.0 - self.cy - self.by

	def __call__(self, t):
		"""
		Calculates the position of a point on the bezier curve at a given time.

		Args:
			t: The normalized time (0.0 to 1.0) along the bezier curve.

		Returns:
			The position of the point on the bezier curve at time t.
		"""
		return ((self.ax * t + self.bx) * t + self.cx) * t, ((self.ay * t + self.by) * t + self.cy) * t

# ---------------------------------------------------------------------

class BezierSegment:
	def __init__(self, x1, y1, x2, y2, cx1, cy1, cx2, cy2):
		"""
		Initializes a bezier segment.

		Args:
			x1: The x coordinate of the start point.
			y1: The y coordinate of the start point.
			x2: The x coordinate of the end point.
			y2: The y coordinate of the end point.
			cx1: The x coordinate of the first control point.
			cy1: The y coordinate of the first control point.
			cx2: The x coordinate of the second control point.
			cy2: The y coordinate of the second control point.
		"""
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.cx1 = cx1
		self.cy1 = cy1
		self.cx2 = cx2
		self.cy2 = cy2
		self._xmin = min(x1, x2, cx1, cx2)
		self._xmax = max(x1, x2, cx1, cx2)
		self._ymin = min(y1, y2, cy1, cy2)
		self._ymax = max(y1, y2, cy1, cy2)

	@property
	def xmin(self):
		''' calculate the minimum and maximum x and y values '''
		return self._xmin

	@property
	def xmax(self):
		return self._xmax

	@property
	def ymin(self):
		return self._ymin

	@property
	def ymax(self):
		return self._ymax

	def __call__(self, x):
		"""
		Calculates the y coordinate of a point on the segment at a given x coordinate.

		Args:
			x: The x coordinate of the point.

		Returns:
			The y coordinate of the point on the segment at x.
		"""
		# normalize x to 0..1 range
		t = (x - self.x1) / (self.x2 - self.x1)
		# evaluate cubic bezier
		bez = CubicBezier(self.cx1, self.cy1, self.cx2, self.cy2)
		bx, by = bez(t)
		return by

class LinearSegment:
		def __init__(self, x1, y1, x2, y2):
			"""
			Initializes a linear segment.

			Args:
				x1: The x coordinate of the start point.
				y1: The y coordinate of the start point.
				x2: The x coordinate of the end point.
				y2: The y coordinate of the end point.
			"""
			self.x1 = x1
			self.y1 = y1
			self.x2 = x2
			self.y2 = y2

		@property
		def xmin(self):
			''' calculate the minimum and maximum x and y values '''
			return min(self.x1, self.x2)

		@property
		def xmax(self):
			return max(self.x1, self.x2)

		@property
		def ymin(self):
			return min(self.y1, self.y2)

		@property
		def ymax(self):
			return max(self.y1, self.y2)


		def __call__(self, x):
			"""
			Calculates the y coordinate of a point on the segment at a given x coordinate.

			Args:
				x: The x coordinate of the point.

			Returns:
				The y coordinate of the point on the segment at x.
			"""
			if self.x2 == self.x1:
				return self.y1
			t = (x - self.x1) / (self.x2 - self.x1)
			return self.y1 + t * (self.y2 - self.y1)

class PiecewiseFunction:
	def __init__(self, *segments):
		self._segments = list(segments)
		self.normalize()

	@property
	def xmin(self):
		''' get the minimum and maximum x and y values '''
		return min([s.xmin for s in self._segments])

	@property
	def xmax(self):
		return max([s.xmax for s in self._segments])

	@property
	def ymin(self):
		return min([s.ymin for s in self._segments])

	@property
	def ymax(self):
		return max([s.ymax for s in self._segments])

	def normalize(self):
		# order segments by x
		self._segments.sort(key=lambda s: s.x1)
		# iterate over segments, finding and fixing gaps and overlaps
		i = 0
		while i < len(self._segments) - 1:
			s1 = self._segments[i]
			s2 = self._segments[i+1]
			# no gap or overlap
			if s1.x2 == s2.x1:
				i = i + 1
				continue
			# gap
			if s1.x2 < s2.x1:
				self._segments.insert(i+1, LinearSegment(s1.x2, s1(s1.x2), s2.x1, s2(s2.x1)))
				i = i + 1
				continue
			# partial overlap
			if s1.x2 > s2.x1:
				xm = (s1.x2 + s2.x1) / 2.0
				s1.x2 = xm
				s2.x1 = xm
				i = i + 1
				continue
			# total overlap - s1 contains s2
			if s1.x1 <= s2.x1 and s1.x2 >= s2.x2:
				# calculate mean y for s2
				mean_y = (s2(s2.x1) + s2(s2.x2)) / 2.0
				# replace s2 with two segments of constant y
				self._segments[i+1:i+2] = [LinearSegment(s2.x1, mean_y, s2.x2, mean_y)]
				i = i + 1
				continue

	def split(self, x):
		# find segment containing x
		for i, s in enumerate(self._segments):
			if x > s.xmin and x < s.xmax:
				if isinstance(s, LinearSegment):
					s1 = LinearSegment(s.x1, s.y1, x, s(x))
					s2 = LinearSegment(x, s(x), s.x2, s.y2)
					self._segments[i:i+1] = [s1, s2]
					return
				elif isinstance(s, BezierSegment):
					# cubic bezier split algorithm from https://stackoverflow.com/a/11086539
					t = (x - s.x1) / (s.x2 - s.x1)
					x11 = s.x1
					y11 = s.y1
					x12 = s.x1 + (s.cx1 - s.x1) * t
					y12 = s.y1 + (s.cy1 - s.y1) * t
					x21 = s.cx2 + (s.x2 - s.cx2) * t
					y21 = s.cy2 + (s.y2 - s.cy2) * t
					x22 = s.x2
					y22 = s.y2
					x10 = (x12 * (1.0-t)) + (x21 * t)
					y10 = (y12 * (1.0-t)) + (y21 * t)
					x01 = (x11 * (1.0-t)) + (x12 * t)
					y01 = (y11 * (1.0-t)) + (y12 * t)
					x02 = (x21 * (1.0-t)) + (x22 * t)
					y02 = (y21 * (1.0-t)) + (y22 * t)
					x00 = (x01 * (1.0-t)) + (x10 * t)
					y00 = (y01 * (1.0-t)) + (y10 * t)
					s1 = BezierSegment(x11, y11, x00, y00, x01, y01, x10, y10)
					s2 = BezierSegment(x00, y00, x22, y22, x10, y10, x02, y02)
					self._segments[i:i+1] = [s1, s2]
					return
		return

	def __call__(self, x):
		# find segment containing x
		for i, s in enumerate(self._segments):
			if x >= s.xmin and x <= s.xmax:
				return s(x)
		return 0

# ---------------------------------------------------------------------
# parsers for animation parameters
# ---------------------------------------------------------------------

def parse_float_or_fraction(s, whole=1.0):
	''' parse a float or fraction string into a float '''
	if '/' in s:
		num, denom = s.strip().split('/')
		return (float(num.strip()) / float(denom.strip())) * whole
	else:
		return float(s)

def parse_piecewise_function(s):
	''' parse a string containing a description of one or more curve segments in the form of 'L x y' or 'C x y a b c d' returning a PiecewiseFunction instance '''
	words = s.strip().split()
	command_args = []
	current_command = []
	for word in words:
		if word.isalpha():
			if len(current_command) > 0:
				command_args.append(current_command)
			current_command = [word]
		else:
			current_command.append(word)
	if len(current_command) > 0:
		command_args.append(current_command)

	segments = []
	for command in command_args:
		if command[0].upper() == 'L':
			# Linear segment: L x y
			x = parse_float_or_fraction(command[1])
			y = parse_float_or_fraction(command[2])
			if len(segments) > 0:
				prev_segment = segments[-1]
				segments.append(LinearSegment(prev_segment.x2, prev_segment.y2, x, y))
			else:
				segments.append(LinearSegment(0, 0, x, y))
		elif command[0].upper() == 'C':
			# Cubic bezier segment: C x y a b c d
			x = parse_float_or_fraction(command[1])
			y = parse_float_or_fraction(command[2])
			cx1 = parse_float_or_fraction(command[3])
			cy1 = parse_float_or_fraction(command[4])
			cx2 = parse_float_or_fraction(command[5])
			cy2 = parse_float_or_fraction(command[6])
			if len(segments) > 0:
				prev_segment = segments[-1]
				segments.append(BezierSegment(prev_segment.x2, prev_segment.y2, x, y, cx1, cy1, cx2, cy2))
			else:
				segments.append(BezierSegment(0, 0, x, y, cx1, cy1, cx2, cy2))
		else:
			raise ValueError("Error in parse_piecewise_function(): '{' '.join(command)}' is not a valid segment")
	return PiecewiseFunction(*segments)

def rescale_time_frames(pwf, beats, fps):
	import math
	pwf = parse_piecewise_function(pwf)
	iframe = 0.0
	lframe = 0.0
	for oframe in range(0, int(beats[-1][0]*fps)):
		bar, beat = time_to_songpos(oframe / fps, beats)
		rate = pwf(math.fmod(bar, 1.0))
		lframe = iframe
		iframe += rate
		yield iframe
