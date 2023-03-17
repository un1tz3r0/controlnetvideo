A CLI tool for applying Stable Diffusion Img2Img with ControlNets to videos, with stabilization provided by feeding back the prior output frame as a partial init image. The input frames are fed in as the ControlNet control image through the appropriate detector.

# Example

Process a video using HED ouline detection...

```
python3 controlnetvideo.py ~/Downloads/PXL_20220309_213229996.TS.mp4 out.mp4 --controlnet hed --steps 20 \
		--prompt 'a cute white cat with black patches watching a snowstorm out a window in a cozy apartment overlooking an intersection during a blizzard in the style of pusheen' \
		--prompt-strength 6 --show-input --show-detector --dump-frames frame.png --init-image-strength 0.55 --feedthrough-strength 0.1 --min-dimension 512 --round-dims-to 128
```

Process the same video using 

```
python controlnetvideo.py \
	~/Downloads/PXL_20220309_213229996.TS.mp4 \
	out-graffuturism.mp4 \
	--controlnet depth \
	--steps 20 \
	--prompt 'graffuturism' \
	--prompt-strength 10 \
	--show-input \
	--show-detector \
	--dump-frames frame.png \
	--init-image-strength 0.35 \
	--feedthrough-strength 0.00 \
	--min-dimension 512 \
	--round-dims-to 128
```

# Usage

```
controlnetvideo.py [OPTIONS] INPUT_VIDEO OUTPUT_VIDEO

Options:
  --start-time UNION              start time in seconds
  --end-time UNION                end time in seconds
  --duration UNION                duration in seconds
  --max-dimension INTEGER         maximum dimension of the video
  --min-dimension INTEGER         minimum dimension of the video
  --round-dims-to INTEGER         round the dimensions to the nearest multiple
                                  of this number
  --prompt TEXT                   prompt used to guide the denoising process
  --negative-prompt TEXT          negative prompt, can be used to prevent the
                                  model from generating certain words
  --prompt-strength FLOAT         how much influence the prompt has on the
                                  output
  --num-inference-steps, --steps INTEGER
                                  number of inference steps, depends on the
                                  scheduler, trades off speed for quality.
                                  20-50 is a good range from fastest to best.
  --controlnet [hed|canny|scribble|openpose|depth|normal|mlsd]
                                  which pretrained controlnet annotator to use
  --controlnet-strength FLOAT     how much influence the controlnet
                                  annotator's output is used to guide the
                                  denoising process
  --fix-orientation               resize videos shot in portrait mode on some
                                  devices to fix incorrect aspect ratio bug
  --init-image-strength FLOAT     the init-image strength, or how much of the
                                  prompt-guided denoising process to skip in
                                  favor of starting with an existing image
  --feedthrough-strength FLOAT    the init-image is composed by transferring
                                  motion from the input video onto each output
                                  frame, and blending it optionally with the
                                  frame input that is sent to the controlnet
                                  detector
  --show-detector                 show the controlnet detector output
  --show-input                    show the input frame
  --show-output                   show the output frame
  --show-flow                     show the motion transfer (not implemented
                                  yet)
  --dump-frames PATH              write frames to a file during processing to
                                  visualise progress. may contain {}
                                  placeholder for frame number
  --help                          Show this message and exit.
```
