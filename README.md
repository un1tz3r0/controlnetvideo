# Stable Diffusion Video2Video ControlNet Model

##### by Victor Condino <un1tz3r0@gmail.com>
##### May 21 2023

This file contains the code for a simple img2img-controlnet video processor, which can apply Stable
Diffusion to a video while attempting to maintain frame-to-frame consistency.	It is based on the
Stable Diffusion img2img model, but adds a motion estimator and motion compensator to
maintain consistency between frames.

### New as of Oct 2024 -- Flux, SDXL Reference

I've added support for **Flux.1** and **SDXL controlnet** and **reference-only** control schemes. See below for some examples...

#### Some examples of sdxl img2img reference

These videos were made with the 'refxl' controlnet option, which is an implementation of reference-only control for sdxl img2img. Effects are interesting. Needs more experimentation.

`./venv/bin/python3 controlnetvideo.py examples/PXL_20240827_063831973.TS.mp4 examples/outh-2.mp4 --prompt kowloon\ walled\ city\ manifold\ garden\ pixel\ perfect\ anton\ fadeev\ studio\ ghibli\ miyazaki\ city\ streets --dump-frames progress.png --show-input --show-output --show-motion --color-info --motion-sigma 0.1 --motion-alpha 0.1</b> --color-fix none --feedthrough-strength 0.08 --swap-images --init-image-strength 0.60 --controlnet refxl`

`./venv/bin/python3 controlnetvideo.py examples/PXL_20240827_063831973.TS.mp4 examples/outh-3.mp4 --prompt manifold\ garden\ cityscape\ billowing\ clouds\ of\ thick\ clored\ smoke\ anton\ fadeev\ studio\ ghibli\ miyazaki\ city\ streets --dump-frames progress.png --show-input --show-output --show-motion --color-info --motion-sigma 0.1 --motion-alpha 0.1</b> --color-fix none <b>--feedthrough-strength 0.2 --swap-images --init-image-strength 0.53 --controlnet refxl`


# Installation

### Pre-requisites

First, clone this repo using git

```sh
git clone https://github.com/un1tz3r0/controlnetvideo.git
cd controlnetvideo
```

You may wish to set up a __venv__. This is not strictly nescessary, so skip this step to use user/system-wide packages at your own risk, may break other projects whose dependencies are out of date.

```sh
python3 -m venv venv
source venv/bin/activate
```

### Dependecies

Now, install the dependencies using pip3:

```sh
pip3 install -r requirements.txt
```

You should now be ready to run the script and process video files. If you are having trouble getting it working, open an issue or reach out on twitter or discord...

## Example 1

To process a video using _Stable Diffusion 2.1_ and a _ControlNet_ trained for __depth-to-image__ generation:

```
python3 controlnetvideo.py \
	PXL_20230422_013745844.TS.mp4 \
	--controlnet depth21 \
	--prompt 'graffuturism colorful intricate heavy detailed outlines' \
	--prompt-strength 9 \
	--show-input \
	--show-detector \
	--show-motion \
	--dump-frames '{instem}_frames/{n:08d}.png' \
	--init-image-strength 0.4 \
	--color-amount 0.3 \
	--feedthrough-strength 0.001 \
	--show-output \
	--num-inference-steps 15 \
	--duration 60.0 \
	--start-time 10.0 \
	--skip-dumped-frames \
	'{instem}_out.mp4'
```

This will process the file `PXL_20230422_013745844.TS.mp4`, starting at `10 seconds` for a duration of `60 seconds`. It will process each input frame with some preprocessing (motion transfer/compensation of the output feedback), followed by a detector and diffusion models in a pipeline configured by the `--controlnet` option. Here, we are using `depth21`, which selects __Midas__ depth estimator for the detector and the __Stable Diffusion 2.1__ model and the matching pretrained __ControlNet__ model, in this case courtesy of [thibaud](https://huggingface.co/thibaud/), for `15` steps the first frame and `(1.0-0.4)*15 => 9` steps (this is because img2img skips initial denoising steps according to the init-image strength) for the remaining frames. The diffusion pipeline will be run with the prompt `'graffuturism colorful intricate heavy detailed outlines'` with a guidance strength of `9`, and full controlnet influence.

During processing, it will show the input, the detector output, the motion estimate, and the output of each frame, by writing them to numbered `.png` image files in a directory `PXL_20230422_013745844.TS_frames/` which will be created if it does not exist. If you just want a single image file you can watch with a viewer which auto-refreshes upon the file changing on disk, then you can specify the filename to `--dump-frames` without a `{n}` substitution, causing it to continually overwrite the same file. This is useful for watching the progress of the video processing in real time. 

![example frame dump showing video processing progress](./examples/00000339.png)

Finally, it will also encode and write the output to a video file `PXL_20230422_013745844.TS_out.mp4`.

https://github.com/un1tz3r0/controlnetvideo/assets/851283/00166ca1-6eb6-46ce-aba9-4e0d019ab31a

### Example 2

Here's another example of the same video, but with a different prompt and different parameters:

```
python3 controlnetvideo.py \
        PXL_20230419_205030311.TS.mp4 \
        --controlnet depth21 \
        --prompt 'mirrorverse colorful intricate heavy detailed outlines' \
        --prompt-strength 10 \
        --show-input \
        --show-detector \
        --show-motion \
        --dump-frames '{instem}_frames/{n:08d}.png' \
        --init-image-strength 0.525 \
        --color-amount 0.2 \
        --feedthrough-strength 0.0001 \
        --show-output \
        --num-inference-steps 16 \
        --skip-dumped-frames \
        --start-time 0.0 \
        '{instem}_out.mp4'
```

The frames dumped will look like this:

![example frame dump](./examples/00001750.png)

And the resulting output video:

https://github.com/un1tz3r0/controlnetvideo/assets/851283/550ff6c7-5be3-4cf0-a25c-603461b8f791


### Tips

- If your video comes out squashed or the wrong aspect ratio, try `--no-fix-orientation` or `--fix-orientation`. You can also mess with the scaling using `--max-dimension` and `--min-dimension` and `--round-dims-to`, although these should all be sane defaults that just work with most sources.

- Feedback strength, set with `--init-image-strength` controls frame-to-frame consistency, by changing how much the motion-compensated previous output frame is fed into the next frame's diffusion pipeline in place of initial latent noise, _a la_ img2img latent diffusion (_citation needed_). Values around 0.3 to 0.5 and sometimes much higher (closer to 1.0, the maximum which means no noise is added and no denoising steps will be run).

- See --help (below) for more options, there are many features not covered here, such as:
  - detector kwargs
  - original input frame feedthrough strength
  - motion estimate spatiotemporal smoothing (crude, simple exponential and gaussian filters, but it can help in some situations give better results if the motion estimate is noisy)
  - color drift correction, and more.
  - more things I forgot to mention

If there is interest, I will write up a more detailed guide to the options and how to use them.

## Usage

```
Usage: controlnetvideo.py [OPTIONS] INPUT_VIDEO OUTPUT_VIDEO

Options:
  --overwrite / --no-overwrite    don't overwrite existing output file -- add
                                  a numeric suffix to get a unique filename.
                                  default: --no-overwrite.
  --start-time FLOAT              start time in seconds
  --end-time FLOAT                end time in seconds
  --duration FLOAT                duration in seconds
  --output-bitrate TEXT           output bitrate for the video, e.g. '16M'
  --output-codec TEXT             output codec for the video, e.g. 'libx264'
  --max-dimension INTEGER         maximum dimension of the video
  --min-dimension INTEGER         minimum dimension of the video
  --round-dims-to INTEGER         round the dimensions to the nearest multiple
                                  of this number
  --fix-orientation / --no-fix-orientation
                                  resize videos shot in portrait mode on some
                                  devices to fix incorrect aspect ratio bug
  --no-audio                      don't include audio in the output video,
                                  even if the input video has audio
  --audio-from PATH               audio file to use for the output video,
                                  replaces the audio from the input video,
                                  will be truncated to duration of input or
                                  --duration if given
  --audio-offset FLOAT            offset in seconds to start the audio from,
                                  when used with --audio-from
  --prompt TEXT                   prompt used to guide the denoising process
  --negative-prompt TEXT          negative prompt, can be used to prevent the
                                  model from generating certain words
  --prompt-strength FLOAT         how much influence the prompt has on the
                                  output
  --num-inference-steps, --steps INTEGER
                                  number of inference steps, depends on the
                                  scheduler, trades off speed for quality.
                                  20-50 is a good range from fastest to best.
  --controlnet [refxl|fluxcanny|depthxl|aesthetic|lineart21|hed|hed21|canny|canny21|openpose|openpose21|depth|depth21|normal|mlsd]
                                  which pretrained model and controlnet type
                                  to use. the default, depthxl, uses the dpt
                                  depth estimator and controlnet with the sdxl
                                  base model
  --controlnet-strength FLOAT     how much influence the controlnet
                                  annotator's output is used to guide the
                                  denoising process
  --init-image-strength FLOAT     the init-image strength, or how much of the
                                  prompt-guided denoising process to skip in
                                  favor of starting with an existing image
  --feedthrough-strength FLOAT    the ratio of input to motion compensated
                                  prior output to feed through to the next
                                  frame
  --motion-alpha FLOAT            smooth the motion vectors over time, 0.0 is
                                  no smoothing, 1.0 is maximum smoothing
  --motion-sigma FLOAT            smooth the motion estimate spatially, 0.0 is
                                  no smoothing, used as sigma for gaussian
                                  blur
  --show-detector / --no-show-detector
                                  show the controlnet detector output
  --show-input / --no-show-input  show the input frame
  --show-output / --no-show-output
                                  show the output frame
  --show-motion / --no-show-motion
                                  show the motion transfer (not implemented
                                  yet)
  --dump-frames PATH              write intermediate frame images to a
                                  file/files during processing to visualise
                                  progress. may contain various {}
                                  placeholders
  --skip-dumped-frames            read dumped frames from a previous run
                                  instead of processing the input video
  --dump-video                    write intermediate dump images to the final
                                  video instead of just the final output image
  --color-fix [none|rgb|hsv|lab]  prevent color from drifting due to feedback
                                  and model bias by fixing the histogram to
                                  the first frame. specify colorspace for
                                  histogram matching, e.g. 'rgb' or 'hsv' or
                                  'lab', or 'none' to disable.
  --color-amount FLOAT            blend between the original color and the
                                  color matched version, 0.0-1.0
  --color-info                    print extra stats about the color content of
                                  the output to help debug color drift issues
  --canny-low-thr FLOAT           canny edge detector lower threshold
  --canny-high-thr FLOAT          canny edge detector higher threshold
  --mlsd-score-thr FLOAT          mlsd line detector v threshold
  --mlsd-dist-thr FLOAT           mlsd line detector d threshold
  --swap-images                   Switch the init and reference images when
                                  using reference-only controlnet
  --help                          Show this message and exit.
```
