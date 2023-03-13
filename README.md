Some experiments with applying Stable Diffusion Controlnet models to video... investigating how to stabilize the output. I have some theories about transferring motion estimation from the original video to the output frames and using that with latent blending as noisy priors...

	`python controlnetvideo.py in.mp4 out.mp4 --prompt 'vaporwave cat licking icedcream' --consistency-strength 0.6 --prompt-strength 8 --controlnet-strength 0.8 --no-fix-orientation`


**Update:** Crap it only worked because I accidentally commented out the latent=noisy_latent line when calling the pipeline! Woops. Still trying to figure out why prepare_latents from img2img returns a different size tensor than prepare_latents from controlnet does... clearly i'm new here :/
