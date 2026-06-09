# TODO — controlnetvideo v0.3.x

Modernize `controlnetvideo.py` to current diffusers models/pipelines, upgrade
motion estimation, and add a flexible, animatable video-filter system that can be
inserted at multiple points in the frame-processing chain.

Reference: up-to-date diffusers clone in `diffusers/`.

---

## 0. Groundwork & understanding

- [ ] Re-read the current frame loop (`frame_filter` in `controlnetvideo.py`) and
      document the existing data flow: input frame → optical flow vs. prev input →
      warp prev output (motion transfer) → blend with input (feedthrough) → detector/
      control image → img2img/controlnet/reference pipeline → color-fix → feedback.
- [ ] Catalogue the existing pipeline branches (`refxl`, `depthxl`, `fluxcanny`,
      legacy SD1.5/2.1 controlnets) and the assumptions each makes about args.
- [ ] Survey `diffusers/` for currently-supported text-to-image + conditioning
      pipelines and note which are worth adopting (see §1).

## 1. Modernize models & pipelines

- [x] Audit available pipelines in the diffusers clone (v0.39-dev): SDXL
      ControlNet-Union (+img2img), SD3/3.5, FLUX (controlnet + structural Control
      + Redux), Qwen-Image, Sana, AnimateDiff V2V/SparseCtrl, IP-Adapter across
      SD/SDXL/SD3/FLUX. (installed runtime diffusers is 0.38-dev — has Union + IP-Adapter.)
- [x] **Phase A (done):** SDXL ControlNet-Union img2img + IP-Adapter feedback.
  - [x] `--controlnet unionxl` + `--control-type depth|canny|openpose|softedge|normal`
        (one Union model, `--union-model` repo override; modes 0/1/2/3/4).
  - [x] `--ip-adapter` / `--ip-adapter-scale` (animatable via `ipadapter`) /
        `--ip-adapter-source feedback|input|output` — injects the motion-warped
        previous output as an image prompt (modern reference-only replacement);
        also wired into the plain `depthxl`/SDXL branch.
  - [x] Validated: branch reached, mode mapping, torch-scoping, CLI binding, compiles.
        NOT yet run on a real GPU (sandbox has no model downloads) — needs a live
        generation smoke test.
- [ ] **Phase B:** FLUX structural Control img2img (Depth/Canny-dev) + Qwen; retire
      the vendored `pipeline_flux_controlnet_image_to_image`; differential-diffusion
      feedback driven by optical-flow occlusion/confidence.
- [ ] **Phase C:** AnimateDiff V2V + ControlNet/SparseCtrl as a parallel temporal engine.
- [ ] SD3 / SD3.5 ControlNets (note: no SD3 controlnet-img2img class — needs adapting).
- [ ] Pin a known-good diffusers version; retire vendored pipeline files as upstream covers them.
- [x] Normalize SDXL/union call sites to a kwargs dict so prompt/strength/control/
      ip-adapter params are passed consistently. (flux/sd2.1 branches still bespoke.)

## 2. Dense motion estimation upgrade — DONE (`motionestimators.py`)

- [x] Pluggable `MotionEstimator` interface (estimate_motion / transfer_motion /
      flow_to_image); estimators swappable via `create_estimator(name)`.
- [x] `--motion-estimator raft|raft_small|farneback|dis`. RAFT (large) is the
      original moved **byte-for-byte** (same transposes / `reverse_order=True`),
      raft_small added; farneback + dis are CPU (no GPU needed).
- [x] `transfer_motion` works per-estimator: RAFT keeps its axis-swapped convention;
      cv2 estimators use the shared standard-convention warp.
- [x] Validated: cv2 estimators on real frames (CPU); RAFT on GPU — and confirmed
      smoothing is a no-op at default params so RAFT output is unchanged.
- [ ] Future: SEA-RAFT / GMFlow / FlowFormer behind the same interface (need
      external weights/repos); image-guided flow filtering needs opencv-contrib.

## 3. Motion smoothing / blending / denoising filters — DONE

- [x] Occlusion-aware temporal smoothing: forward/backward consistency (adaptive
      Sundaram criterion) suppresses flow carry-over at (dis)occlusions
      (`--motion-occlusion`; standard-convention estimators only — gated+warned for raft).
- [x] Edge-preserving spatial smoothing: `--motion-spatial-filter gaussian|bilateral`
      (per-channel bilateral on the flow keeps motion boundaries; gaussian = legacy).
- [x] `smooth_flow_spatiotemporal` realised as a **causal** windowed average
      (`--motion-temporal-window N`, gaussian-in-time weights) combined with the
      spatial filter — no lookahead, so no added latency.
- [ ] Make motion-filter parameters animatable (see §5) — currently set at
      construction, not per-frame.
- [ ] Future: image-guided (joint-bilateral/guided) flow smoothing once
      opencv-contrib/`ximgproc` is available; expose the occlusion mask to weight
      the feedback blend (ties into Phase B differential diffusion).

## 4. Generic video-filter system

Implemented as a standalone, independently-testable module: `videofilters.py`
(parsing + filters + its own CLI). Imported into `controlnetvideo.py` only once
solid — see the integration items at the end of this section.

- [x] Implement `parse_video_filter_spec(spec) -> callable` that returns a
      stateful callable (`FilterChain`) applied successively to frames:
  - [x] Spec mini-language, audio-animate style: `;` between filters, `,` between
        params, `=` assigns; values are constants or `L/C` animation curves.
  - [x] Support stateful temporal filters (chain retains per-filter state across
        calls) and stateless spatial filters.
  - [x] Per-frame curve evaluation at a position `x` passed in by the caller.
- [x] Standalone CLI in `videofilters.py` (apply a chain to an input video → output
      video; `--list-filters`, clip/`time`/`--audio-from` curve positions). Tested
      end-to-end on a clip; animation + overlays confirmed visually.
- [x] Implement the filter library:
  - [x] **Blur / denoise / sharpen** — `blur`, `boxblur`, `bilateral`,
        `edgepreserve`, `nlmeans`, `tdenoise` (temporal), `sharpen`.
  - [x] **Color modifications** — `hsl`, `hueshift`, `saturation`, `brightness`,
        `contrast`, `gamma`.
  - [x] **Geometric distortions** (origin + velocity, time-accumulated) — `pan`,
        `zoom`, `spin`, with shared `edge` (wrap/reflect/clamp/black) handling.
  - [x] **Depth/detector region effects** — `lineart` overlay + `depthfx`
        (effect=brighten|darken|blur|outline|inner/outer shadow|inner/outer blur;
        region=fg|bg; animatable threshold/feather/amount). Lazy `DPTDepthProvider`;
        depth backend is injectable.
  - [x] **Animatable pattern overlays** — single `pattern` filter, shared generator:
        stripes, dashed, zigzag, chevron, checker, grid, polkadot, honeycomb;
        common params scale/angle/speed/duty/color1/color2/blend/amount.
  - [x] Blend modes: normal, multiply, screen, overlay, softlight, add, subtract,
        difference, lighten, darken.
- [ ] Follow-ups / polish for the filter module:
  - [ ] Guided-filter / domain-transform variant for edge-aware denoise (currently
        bilateral + `edgePreservingFilter`).
  - [ ] Rolling-window / motion-aware temporal denoise (currently exponential only).
  - [ ] Optional non-DPT depth backend + alternate detectors for `lineart`
        (HED/lineart via controlnet_aux).
  - [ ] Animatable color params (curves on hex colors) if wanted.
- [x] **Integrate into `controlnetvideo.py`**:
- [x] Add CLI options to attach a filter chain at each chain point:
  - [x] `--input-filter`        — raw input frame.
  - [x] `--feedforward-filter`  — input contribution into the next-frame init.
  - [x] `--motion-filter`       — frames fed to the motion estimator (own
        `prev_motion_frame` stream so both compared frames are filtered alike).
  - [x] `--control-filter`      — control/conditioning image.
  - [x] `--feedback-filter`     — warped prev-output before blending back in.
  - [x] `--output-filter`       — final output frame (also what is fed back).
- [x] Wire each filter chain into the correct location in `frame_filter`; chains
      parsed once up front (fast-fail on bad spec) and run at the same animation
      position `x` as `--audio-animate` (also fixed the no-audio `x` path).
- [x] Each filter point is its own `FilterChain` instance, holding independent
      per-filter state across frames.

## 4b. Beat-extraction enhancements (`extract_beats.py`)

- [x] Cache beat analysis to a JSONL file (`<source>.jsonl`): metadata header line
      plus one line per beat with `time` (seconds), `beat`, and `frame@fps` when an
      fps is supplied.
- [x] Per-beat / per-bar RMS amplitude (`beat_rms`, `bar_rms`) over the trailing
      beat- and bar-length windows — an intensity / buildup-drop measure (cached;
      not yet consumed by the pipeline).
- [x] Reuse the cache when present + valid (matching crop & samplerate), skipping the
      slow RNN/DBN analysis; stale or unreadable caches re-analyse and overwrite.
- [x] Progress bar for analysis: live bar over the threaded RNN/DBN downbeat tracker
      plus a determinate bar over the RMS pass.
- [x] Backward-compatible: `extract_beats()` still returns `(audio, sig, beats)`;
      rich entries exposed via new `extract_beat_data()`. Callers pass `fps`.
- [ ] Follow-up: actually use `beat_rms`/`bar_rms` to drive effect punch-in/out or
      scene-change edits (e.g. expose as an animation source).

## 5. Animation-curve integration

- [ ] Extend the animatable-parameter system (currently `denoise`, `feedthrough`,
      `control`, `guidance`) so video-filter and motion-filter params can be
      animated via the same curve/`--audio-animate` mechanism.
- [ ] Define a naming scheme to address a specific filter's specific param
      (e.g. `input.blur.sigma`) from the animation spec.
- [ ] Pass per-frame resolved param values into each filter callable.
- [ ] Update `extract_beats.parse_piecewise_function` usage / docs as needed.

## 6. CLI, docs, and cleanup — mostly DONE

- [x] README updated: new features list, video-filter language + insertion-point table,
      examples; fresh `--help` block; legacy note. Example 1/2 updated off the removed
      `depth21` onto `depthxl`.
- [x] Example invocations showing filter chains at each insertion point (README).
- [x] Removed dead/legacy code: the broken SD1.5/SD2.1 `--controlnet` branches
      (canny/depth/openpose/hed/mlsd/normal/aesthetic/*21) and now-dead helpers
      (`MidasDetectorWrapper`, `depth_to_normal`), duplicate `hsltorgb`, the commented
      Farneback / `smooth_flow_spatiotemporal` blocks, unreachable legacy pipeline +
      pipe-call `else` branches, dead `scheduler`/`sdmodel` vars, stale commented imports.
      `--controlnet` is now `unionxl|refxl|fluxcanny|depthxl`; mlsd options removed.
      (controlnetvideo.py ~1275 → ~1023 lines.)
- [x] `.gitignore`: `*TEMP_MPY_*`, `progress*.png`, `out*.mp4`, `*.jsonl`.
- [~] Smoke-test: filter insertion points (parse + wiring) and motion estimators
      (raft on GPU, farneback/dis on CPU) validated. Full per-model diffusion smoke
      test still needs a networked GPU (sandbox can't download base models).

## 7. Stretch / nice-to-have

- [ ] Visualization/debug dumps for filtered intermediates (extend `--dump-frames`).
- [ ] Benchmark motion estimators and filters for speed/quality.
- [ ] Consider true temporal-aware video diffusion models as a separate mode.

## 8. Sequenced prompts, unified time units & filter-chain grouping (future / not started)

> **Deferred** — parked here for later; do the unfinished items above first. Captures a
> bigger language/grammar expansion across the prompt and filter systems.

> Original request:
> *"Extend the animation system to allow specifying multiple prompts with durations for
> each, which it loops through, with a period of the total length of all prompts. Syntax:
> `<prompt>:<duration>;<prompt>:<duration>;...` where prompt is any string without a
> semicolon or colon (backslash escapes those and any other single character, interpreted
> literally), and duration is a decimal or `numerator/denominator` fraction followed by one
> of `s`/seconds, `b`/bars, `f`/frames, `t`/total, the units to multiply by. Extend this
> time format to the rest of the animation system wherever times are specified. Add a
> grouping syntax to the filter spec: anonymous timed groups `{filters...} 4b; {filters...}
> 1b;`, named groups `@name {filters};`, and references `$name` inside a block that splice
> the named group's filters inline — purely a convenience."*

### 8a. Unified time-quantity format
- [ ] `parse_time(spec, ctx) -> seconds`: quantity (decimal or `n/d` fraction) × unit
      (`s`/`seconds`, `b`/`bars`, `f`/`frames`, `t`/`total`).
- [ ] Resolution context carries fps, total clip duration, and the beat/bar map (from
      `extract_beats`); each frame already knows its time/`x`.
- [ ] Define behaviour when a unit's basis is unavailable: `b`/bars without `--audio-from`
      (error, or derive from a declared/default BPM?), `t`/total before duration is known.
- [ ] Adopt the format everywhere times are specified (`--start-time`/`--end-time`/
      `--duration`, etc.), keeping bare-seconds inputs backward compatible.

### 8b. Multi-prompt sequencing
- [ ] Grammar `<prompt>:<duration>; ...` with backslash escaping (`\;`, `\:`, `\\`,
      `\<any>` → literal char).
- [ ] Loop with period = sum of durations; map current time → active prompt.
- [ ] Wire into `--prompt` (and `--negative-prompt`) so the pipeline prompt varies per
      frame; single-prompt usage stays valid.
- [ ] Cache encoded prompt embeddings per distinct prompt (avoid re-encoding every frame).
- [ ] (Stretch) optional crossfade of prompt embeddings across boundaries instead of a
      hard switch.

### 8c. Filter-chain grouping & time sequencing
- [ ] Anonymous timed groups in a filter spec: `{ ...filters... } <duration>; { ... }
      <duration>;` sequenced and looped (period = sum of durations).
- [ ] A bare/un-timed chain stays equivalent to today's behaviour.
- [ ] Select the active group for the current time within the period; preserve per-filter
      state correctly across group switches (define reset vs. carry-over semantics).

### 8d. Named groups & references (macro expansion)
- [ ] Definition `@name { ...filters... };` (defines, applies nothing on its own).
- [ ] Reference `$name` inside a block splices the named group's filters inline
      (structural copy at parse time).
- [ ] Detect undefined names and reference cycles; clear errors.

### 8e. Parser / architecture refactor
- [ ] Replace the flat `;`/`,` splitting with a small tokenizer + recursive-descent
      parser handling escaping, braces, `@`/`$`, and durations.
- [ ] Reuse the shared time-format parser (8a) across prompt sequencing (8b) and filter
      grouping (8c).
- [ ] Thread the resolution context (fps, total, beats, current time/`x`) through chain
      evaluation.

### 8f. Integration & docs
- [ ] Expose in both `controlnetvideo.py` and the standalone `videofilters.py` CLI.
- [ ] Update `--help` / README with the grammar and worked examples.
- [ ] Escaping/round-trip + sequencing tests; verify backward compatibility with existing
      simple prompt and filter specs.
