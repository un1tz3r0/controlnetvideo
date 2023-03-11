Some experiments with applying Stable Diffusion Controlnet models to video... investigating how to stabilize the output. I have some theories about transferring motion estimation from the original video to the output frames and using that with latent blending as noisy priors...

*Update:* So yeah, it seems to work pretty well... we use dense optical flow methods from OpenCV to estimate the motion between successive input frames, and then warp the prior output frame accordingly, then encode that into the duffusion model's latent space using the pipeline's VAE, similar to how img2img works (in fact that code is taken almost verbatim from the inpainting pipeline).

