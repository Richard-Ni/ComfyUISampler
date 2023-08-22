import torch

import os
import sys
import numpy as np
import latent_preview

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.model_management
import comfy.sample

def prepare_noise(latent_image, seed, noise_inds=None, device="cpu"):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    if device == "cpu":
        generator = torch.manual_seed(seed)
    else:
        generator = torch.cuda.manual_seed(seed)

    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                           generator=generator, device=device)

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout,
                            generator=generator, device=device)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


def common_ksampler(model, source, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    if source == "CPU":
        device = "cpu"
    else:
        device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = prepare_noise(latent_image, seed, batch_inds, device=device)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = latent_preview.get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )


class KSamplerPro:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "source":(["CPU", "GPU"], {"default": "GPU"}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, source, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return common_ksampler(model, source, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)


NODE_CLASS_MAPPINGS = {
    "RN_KSamplerPro": KSamplerPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RN_KSamplerPro": "KSamplerPro",
}
