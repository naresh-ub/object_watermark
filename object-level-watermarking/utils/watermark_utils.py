import datetime
import functools
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.utils import is_wandb_available
from PIL import Image

# Remember that the path is in the w_AIGC folder
from pytorch_fid.fid_score import (
    InceptionV3,
    calculate_frechet_distance,
    compute_statistics_of_path,
)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets.folder import default_loader, is_image_file
from tqdm import tqdm

from loss.loss_provider import LossProvider
from utils.hidden_msg_decoder import get_hidden_decoder, get_hidden_decoder_ckpt
from utils.diffusers_imports import *

device = "cuda"

def iterative_denoising(
    timesteps,
    noisy_latent,
    scheduler,
    unet,
    text_embeddings,
    text_embeddings_f,
    null_text_embeddings,
    denoise_timesteps=None,
    mode = "watermark",
):
    first_noise_pred = None
    latents = []

    if denoise_timesteps is None:
        denoise_timesteps = float(timesteps.cpu().detach().numpy()[0])

    ts = torch.arange(denoise_timesteps, 0, -1)
    # print("Denoising timesteps: ", ts)

    for i, t in enumerate(ts):
        t = int(t)
        # print("Denoising timestep: ", t)
        # Expand latents by 2 if doing classifier-free guidance to do parallel processing
        latent_model_input = torch.cat([noisy_latent] * 2)
        # sigma = scheduler.sigmas[i] ff

        # Scale latents (preconditioning)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict noise residual by passing text embeddings through unet
        # with torch.no_grad():
        
        if t > 10:
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat(
                    [null_text_embeddings, text_embeddings_f], dim=0
                ),
            ).sample
        else:
            # print("using watermark embeddings")
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat(
                    [null_text_embeddings, text_embeddings], dim=0
                ),
            ).sample
            
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        if first_noise_pred is None:
            first_noise_pred = noise_pred_text

        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

        # Compute previous noisy sample
        noisy_latent = scheduler.step(noise_pred, t, noisy_latent).prev_sample
        latents.append(noisy_latent)

    return noisy_latent, first_noise_pred, latents

def iterative_denoising_for_NTI(
    timesteps,
    latents,
    scheduler,
    unet,
    text_embeddings,
    text_embeddings_f,
    null_text_embeddings,
    denoise_timesteps=None,
    mode = "watermark",
):
    
    if mode == "watermark":
        first_noise_pred = None

        if denoise_timesteps is None:
            denoise_timesteps = float(timesteps.cpu().detach().numpy()[0])

        ts = torch.arange(denoise_timesteps, 0, -1)

        denoised_latents = []

        for i, t in enumerate(ts):
            t = int(t)
            
            # Expand latents by 2 if doing classifier-free guidance to do parallel processing
            latent_model_input = torch.cat([latents] * 2)

            # Scale latents (preconditioning)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat(
                    [null_text_embeddings, text_embeddings], dim=0
                ),
            ).sample
                
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            if first_noise_pred is None:
                first_noise_pred = noise_pred_text

            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            denoised_latents.append(latents)
            
    else:
        first_noise_pred = None

        if denoise_timesteps is None:
            denoise_timesteps = float(timesteps.cpu().detach().numpy()[0])

        ts = torch.arange(denoise_timesteps, 0, -1)

        denoised_latents = []

        for i, t in enumerate(ts):
            t = int(t)
            
            # Expand latents by 2 if doing classifier-free guidance to do parallel processing
            latent_model_input = torch.cat([latents] * 2)

            # Scale latents (preconditioning)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat(
                    [null_text_embeddings, text_embeddings_f], dim=0
                ),
            ).sample
                
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            if first_noise_pred is None:
                first_noise_pred = noise_pred_text

            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            denoised_latents.append(latents)

    return latents, first_noise_pred, denoised_latents


def iterative_denoising_gt(
    latents,
    scheduler,
    unet,
    text_embeddings,
    null_text_embeddings,
    denoise_timesteps=None,
):
    # Set denoise steps based on denoise_timesteps or use a default value
    if denoise_timesteps is None:
        raise ValueError("denoise_timesteps must be provided")

    first_noise_pred = None

    scheduler.set_timesteps(50)

    # Denoising process
    for i, t in tqdm(
        enumerate(scheduler.timesteps),
    ):
        # Expand latents by 2 for classifier-free guidance and parallel processing
        latent_model_input = torch.cat([latents] * 2)

        # Scale latents (preconditioning)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict noise residual by passing text embeddings through the UNet
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=torch.cat(
                [null_text_embeddings, text_embeddings], dim=0
            ),
        ).sample

        # Store the first noise prediction for comparison
        if first_noise_pred is None:
            first_noise_pred = noise_pred

        # Perform classifier-free guidance (CFG)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

        # Compute the previous noisy sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # The final denoised latents are the result after completing the loop
    del latent_model_input, noise_pred, noise_pred_uncond, noise_pred_text
    torch.cuda.empty_cache()

    return latents, first_noise_pred


def get_psnr(input_img, decoded_img, range):
    return peak_signal_noise_ratio(input_img, decoded_img, data_range=range)


def get_ssim(input_img, decoded_img):
    return structural_similarity(
        input_img, decoded_img, data_range=1.0, channel_axis=2
    )  # expects in [H, W, C]


normalize_vqgan = transforms.Normalize(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
)  # Normalize (x - 0.5) / 0.5
unnormalize_vqgan = transforms.Normalize(
    mean=[-1, -1, -1], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)  # Unnormalize (x * 0.5) + 0.5
normalize_img = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)  # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)  # Unnormalize (x * std) + mean


# variables and constants
vqgan_transform_256 = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize_vqgan,
    ]
)  # takes a PIL image and returns a tensor of shape (C, 256, 256) in the range [-2, 2]

vqgan_transform_512 = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        normalize_vqgan,
    ]
)  # takes a PIL image and returns a tensor of shape (C, 512, 512) in the range [-2, 2]

vqgan_to_imnet_256 = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        unnormalize_vqgan,
        normalize_img,
    ]
)  # takes a tensor in the range [-2, 2] and returns a tensor of shape (C, 256, 256) in the range [0, 1]

vqgan_to_imnet_512 = transforms.Compose(
    [
        # transforms.Resize(512),
        # transforms.CenterCrop(args.hidden_resolution),
        unnormalize_vqgan,
        normalize_img,  # NOTE: Normalizes to imnet
    ]
)  # takes a tensor in the range [-2, 2] and returns a tensor in the range [0, 1]

## Finetuning hidden msg_decoder


@functools.lru_cache()
def get_image_paths(path):
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return sorted([fn for fn in paths if is_image_file(fn)])


class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

    def __init__(self, path, transform=None, loader=default_loader):
        self.samples = get_image_paths(path)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img = self.loader(self.samples[idx])
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    """Collate function for data loader. Allows to have img of different size"""
    return batch


def get_dataloader(
    data_dir,
    transform,
    batch_size=128,
    num_imgs=None,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
):
    """Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/..."""
    dataset = ImageFolder(data_dir, transform=transform)
    if num_imgs is not None:
        dataset = Subset(
            dataset, np.random.choice(len(dataset), num_imgs, replace=False)
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )


def msg2str(msg):
    return "".join([("1" if el else "0") for el in msg])


def str2msg(str):
    return [True if el == "1" else False for el in str]


def return_msg_decoder(args, accelerator, whiten_save_path):
    print(f">>> Building hidden decoder with weights from {args.msg_decoder_path}...")
    if "torchscript" in args.msg_decoder_path:
        msg_decoder = torch.jit.load(args.msg_decoder_path).to(accelerator.device)
        # already whitened
    else:
        print(">>> Whitening")
        print(
            args.msg_decoder_path,
            args.num_bits,
            args.redundancy,
            args.decoder_depth,
            args.decoder_channels,
        )
        msg_decoder = get_hidden_decoder(
            num_bits=args.num_bits,
            redundancy=args.redundancy,
            num_blocks=args.decoder_depth,
            channels=args.decoder_channels,
        ).to(accelerator.device)
        ckpt = get_hidden_decoder_ckpt(args.msg_decoder_path)
        print(msg_decoder.load_state_dict(ckpt, strict=False))
        msg_decoder.eval()

        with torch.no_grad():
            # features from the dataset
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            loader = get_dataloader(
                args.hidden_whiten_dir, transform, batch_size=16, collate_fn=None
            )
            ys = []
            for i, x in enumerate(loader):
                x = x.to(accelerator.device)  # normalize_img transform [-2, 2]
                y = msg_decoder(x)
                ys.append(y.to("cpu"))
            ys = torch.cat(ys, dim=0)
            nbit = ys.shape[1]

            # whitening
            mean = ys.mean(dim=0, keepdim=True)  # NxD -> 1xD
            ys_centered = ys - mean  # NxD
            cov = ys_centered.T @ ys_centered
            e, v = torch.linalg.eigh(cov)
            L = torch.diag(1.0 / torch.pow(e, exponent=0.5))
            weight = torch.mm(L, v.T)
            bias = -torch.mm(mean, weight.T).squeeze(0)
            linear = nn.Linear(nbit, nbit, bias=True)
            linear.weight.data = np.sqrt(nbit) * weight
            linear.bias.data = np.sqrt(nbit) * bias
            msg_decoder = nn.Sequential(msg_decoder, linear.to(accelerator.device))
            torchscript_m = torch.jit.script(msg_decoder)

            print(f">>> Creating torchscript at {whiten_save_path}...")
            torch.jit.save(torchscript_m, whiten_save_path)

    return msg_decoder


def return_losses(args, accelerator):
    if args.loss_w == "mse":
        loss_w = lambda decoded, keys, temp=10.0: torch.mean(
            (decoded * temp - (2 * keys - 1)) ** 2
        )  # b k - b k
    elif args.loss_w == "bce":
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(
            decoded * temp, keys, reduction="mean"
        )
    else:
        raise NotImplementedError

    if args.loss_i == "mse":
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs) ** 2)
    elif args.loss_i == "watson-dft":
        provider = LossProvider()
        loss_percep = provider.get_loss_function(
            "Watson-DFT", colorspace="RGB", pretrained=True, reduction="sum"
        )
        loss_percep = loss_percep.to(accelerator.device)
        loss_i = (
            lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0)
            / imgs_w.shape[0]
        )
    elif args.loss_i == "watson-vgg":
        provider = LossProvider()
        loss_percep = provider.get_loss_function(
            "Watson-VGG", colorspace="RGB", pretrained=True, reduction="sum"
        )
        loss_percep = loss_percep.to(accelerator.device)
        loss_i = (
            lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0)
            / imgs_w.shape[0]
        )
    elif args.loss_i == "ssim":
        provider = LossProvider()
        loss_percep = provider.get_loss_function(
            "SSIM", colorspace="RGB", pretrained=True, reduction="sum"
        )
        loss_percep = loss_percep.to(accelerator.device)
        loss_i = (
            lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0)
            / imgs_w.shape[0]
        )
    else:
        raise NotImplementedError

    return loss_w, loss_i

def return_bit_acc_train(keys, latents, vae, msg_decoder):
    imgs_w = vae.decode(latents / vae.config.scaling_factor)[0]
    decoded = msg_decoder(vqgan_to_imnet_512(imgs_w))
    
    diff = ~torch.logical_xor(decoded > 0, keys > 0)  # b k -> b k
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
    
    return bit_accs.mean().item()
    
## Attacks for validation

import numpy as np
from augly.image import functional as aug_functional
import torch
from torchvision import transforms
from torchvision.transforms import functional

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.center_crop(x, new_edges_size)

def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)

def rotate(x, angle):
    """ Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)

def adjust_brightness(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))

def adjust_contrast(x, contrast_factor):
    """ Adjust contrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

def adjust_saturation(x, saturation_factor):
    """ Adjust saturation of an image
    Args:
        x: PIL image
        saturation_factor: saturation factor
    """
    return normalize_img(functional.adjust_saturation(unnormalize_img(x), saturation_factor))

def adjust_hue(x, hue_factor):
    """ Adjust hue of an image
    Args:
        x: PIL image
        hue_factor: hue factor
    """
    return normalize_img(functional.adjust_hue(unnormalize_img(x), hue_factor))

def adjust_gamma(x, gamma, gain=1):
    """ Adjust gamma of an image
    Args:
        x: PIL image
        gamma: gamma factor
        gain: gain factor
    """
    return normalize_img(functional.adjust_gamma(unnormalize_img(x), gamma, gain))

def adjust_sharpness(x, sharpness_factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return normalize_img(functional.adjust_sharpness(unnormalize_img(x), sharpness_factor))

def overlay_text(x, text='Lorem Ipsum'):
    """ Overlay text on image
    Args:
        x: PIL image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.overlay_text(pil_img, text=text))
    return normalize_img(img_aug)

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: PIL image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)
