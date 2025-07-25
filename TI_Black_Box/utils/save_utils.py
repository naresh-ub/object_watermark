import math
import os

import matplotlib.pyplot as plt
import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.watermark_utils import unnormalize_img, unnormalize_vqgan


def save_progress(
    text_encoder,
    placeholder_token_ids,
    accelerator,
    args,
    save_path,
    safe_serialization=True,
):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(
            learned_embeds_dict, save_path, metadata={"format": "pt"}
        )
    else:
        torch.save(learned_embeds_dict, save_path)


def normalize_image(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Rescale to [0, 1]
    return img


def save_comparison_image(
    input_images, decoded_images, output_dir, epoch, step, label=None, img_metrics=None
):
    input_images = normalize_image(input_images)
    decoded_images = normalize_image(decoded_images)

    """Plot input and decoded images side by side and save the figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the input image
    axes[0].imshow(input_images)
    axes[0].set_title("Input Image")
    axes[0].axis("off")  # Hide axes

    # Display the decoded image
    axes[1].imshow(decoded_images)
    axes[1].set_title("Noised Image")
    axes[1].axis("off")  # Hide axes

    if img_metrics != None:  # implement this for later
        plt.suptitle(
            f"PSNR: {img_metrics['psnr']:.4f}, SSIM: {img_metrics['ssim']:.4f}"
        )

    save_path = os.path.join(output_dir, f"comparison_{label}_{epoch}_step_{step}.png")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison figure to {save_path}")


def save_comparison_image_3(
    imgs_d0,
    imgs_w,
    output_dir,
    epoch,
    step,
    label=None,
    img_metrics=None,
):
    imgs_d0 = normalize_image(imgs_d0)
    imgs_w = normalize_image(imgs_w)
    diff_img = np.abs(imgs_d0 - imgs_w)

    """Plot input and decoded images side by side and save the figure."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    # Display the input image
    axes[0].imshow(imgs_d0)
    axes[0].set_title("No watermark")
    axes[0].axis("off")  # Hide axes

    # Display the decoded image
    axes[1].imshow(imgs_w)
    axes[1].set_title("Watermarked")
    axes[1].axis("off")  # Hide axes

    # Display the decoded image
    axes[2].imshow(diff_img)
    axes[2].set_title("Diff Image")
    axes[2].axis("off")  # Hide axes

    if img_metrics != None:  # implement this for later
        if "step" in img_metrics:
            plt.suptitle(
                f"PSNR: {img_metrics['psnr']:.4f}, T: {img_metrics['timesteps']}, Train Step: {img_metrics['step']} Bit_acc: {img_metrics['bit_acc']:.4f}"
            )
        else:
            plt.suptitle(
                f"PSNR: {img_metrics['psnr']:.4f} T: {img_metrics['timesteps']:.4f} Bit_acc: {img_metrics['bit_acc']:.4f}"
            )

    save_path = os.path.join(output_dir, f"comparison_{label}.png")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # print(f"Saved comparison figure to {save_path}")
    
def save_comparison_image_10(
    vae,  # The VAE model used for decoding latents
    nw_latents,  # List of 10 latents without W
    w_latents,  # List of 10 latents with W
    output_dir,  # Directory to save the image
    epoch,  # Epoch number for naming the file
    step,  # Step number for naming the file
    label=None,  # Label for naming the file
):
    # Create a figure with 2 rows and 10 columns for side-by-side comparison
    fig, axes = plt.subplots(2, len(w_latents), figsize=(25, 6))
    
    for i in range(len(w_latents)):
        # Decode each latent in the list using the VAE
        nw_latent_decoded = vae.decode(nw_latents[i] / vae.config.scaling_factor)[0]
        w_latent_decoded = vae.decode(w_latents[i] / vae.config.scaling_factor)[0]

        # Convert torch tensors to numpy arrays
        nw_latent_decoded = nw_latent_decoded.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        w_latent_decoded = w_latent_decoded.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        # Normalize the images for display (optional, depending on image range)
        nw_latent_decoded = normalize_image(nw_latent_decoded)
        w_latent_decoded = normalize_image(w_latent_decoded)

        # Plot the decoded latents without W (top row)
        axes[0, i].imshow(nw_latent_decoded)
        axes[0, i].set_title(f"Latent {len(w_latents)-i} NW")
        axes[0, i].axis("off")  # Hide axes

        # Plot the decoded latents with W (bottom row)
        axes[1, i].imshow(w_latent_decoded)
        axes[1, i].set_title(f"Latent {len(w_latents)-i} W")
        axes[1, i].axis("off")  # Hide axes

    # Set the title for the entire figure
    plt.suptitle(f"Latent Comparison: NW vs W for {label}, Epoch: {epoch}, Step: {step}")

    # Save the figure
    save_path = os.path.join(output_dir, f"comparison_10_latents_{label}_{epoch}_step_{step}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved comparison figure to {save_path}")

def plot_latent_norms(nw_latents, w_latents, output_dir, epoch, step, label=None):
    """
    This function calculates the L2 norm of each latent in the list and 
    plots them as a time series for both nw_latents and w_latents.
    """

    # Calculate L2 norm for each latent in nw_latents and w_latents
    nw_norms = [torch.norm(latent).item() for latent in nw_latents]
    w_norms = [torch.norm(latent).item() for latent in w_latents]

    # Create a figure for the plot
    plt.figure(figsize=(10, 6))

    # Plot norms of nw_latents
    plt.plot(range(len(nw_latents)), nw_norms, label="Norm of nw_latents", marker="o")
    
    # Plot norms of w_latents
    plt.plot(range(len(w_latents)), w_norms, label="Norm of w_latents", marker="o")

    # Add labels and title
    plt.xlabel("Latent Index (Time Step)")
    plt.ylabel("L2 Norm")
    plt.title(f"Latent Norms as Time Series: {label}, Epoch: {epoch}, Step: {step}")

    # Add a legend
    plt.legend()

    # Save the plot
    save_path = os.path.join(output_dir, f"latent_norms_{label}_{epoch}_step_{step}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved latent norms plot to {save_path}")
    
def save_comparison_image_3_SS(
    input_images,
    decoded_images,
    decoded_images_w,
    output_dir,
    epoch,
    step,
    label=None,
    img_metrics=None,
):
    input_images = normalize_image(input_images)
    decoded_images = normalize_image(decoded_images)
    decoded_images_w = normalize_image(decoded_images_w)

    """Plot input and decoded images side by side and save the figure."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    # Display the input image
    axes[0].imshow(input_images)
    axes[0].set_title("Input Image")
    axes[0].axis("off")  # Hide axes

    # Display the decoded image
    axes[1].imshow(decoded_images)
    axes[1].set_title("Denoised, no W")
    axes[1].axis("off")  # Hide axes

    # Display the decoded image
    axes[2].imshow(decoded_images_w)
    axes[2].set_title("Denoised, W")
    axes[2].axis("off")  # Hide axes

    if img_metrics != None:  # implement this for later
        if "step" in img_metrics:
            plt.suptitle(
                f"PSNR: {img_metrics['psnr']:.4f}, Train Step: {img_metrics['step']} Bit_acc: {img_metrics['bit_acc']:.4f}"
            )
        else:
            plt.suptitle(
                f"PSNR: {img_metrics['psnr']:.4f}, Bit_acc: {img_metrics['bit_acc']:.4f}"
            )

    save_path = os.path.join(output_dir, f"comparison_{label}_{epoch}_step_{step}.png")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # print(f"Saved comparison figure to {save_path}")


def get_psnr(x, y, img_space="vqgan"):
    """
    Return PSNR
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == "vqgan":
        delta = torch.clamp(unnormalize_vqgan(x), 0, 1) - torch.clamp(
            unnormalize_vqgan(y), 0, 1
        )
    elif img_space == "img":
        delta = torch.clamp(unnormalize_img(x), 0, 1) - torch.clamp(
            unnormalize_img(y), 0, 1
        )
    else:
        delta = x - y
    delta = 255 * delta
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    psnr = 20 * np.log10(255) - 10 * torch.log10(
        torch.mean(delta**2, dim=(1, 2, 3))
    )  # B
    return psnr


def get_ssim(input_img, decoded_img):
    return structural_similarity(
        input_img, decoded_img, data_range=1.0, channel_axis=2
    )  # expects in [H, W, C]


def normalize_to_0_1(image):
    # Ensure the image values are between -1 and 1
    assert (
        np.min(image) >= -1 and np.max(image) <= 1
    ), "Image values are not in the range [-1, 1]"

    # Normalize image values from [-1, 1] to [0, 1]
    normalized_image = (image + 1) / 2
    return normalized_image


def decode_images_and_plot(
    noise_scheduler,
    timesteps,
    vae,
    noisy_latents,
    batch,
    model_pred,
    epoch,
    global_step,
    args,
    label=None,
    img_metrics=None,
    img_transforms=None,
):
    denoised_latents = noise_scheduler.step(
        model_pred, timesteps, noisy_latents
    ).pred_original_sample

    decoded_images = vae.decode(
        denoised_latents / vae.config.scaling_factor
    ).sample  # values between -1 to 1

    if img_transforms != None:
        decoded_images = img_transforms(decoded_images)

    decoded_images_np = (
        decoded_images.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    )
    input_images_np = (
        batch["pixel_values"].squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    )

    # decoded_imgs_plot = decoded_images_np
    # input_imgs_plot = input_images_np

    decoded_imgs_plot = Image.fromarray(
        np.clip((decoded_images_np) * 255, 0, 255).astype(np.uint8)
    )
    input_imgs_plot = Image.fromarray(
        np.clip((input_images_np) * 255, 0, 255).astype(np.uint8)
    )

    if img_metrics != None:
        img_metrics = {
            "psnr": get_psnr(input_images_np, decoded_images_np, range=1),
            "ssim": get_ssim(input_images_np, decoded_images_np),
        }

    save_comparison_image(
        input_imgs_plot,
        decoded_imgs_plot,
        args.output_dir,
        epoch,
        global_step,
        label=label,
        img_metrics=img_metrics,
    )

    return decoded_images, img_metrics


def save_progress(
    text_encoder,
    placeholder_token_ids,
    accelerator,
    args,
    save_path,
    safe_serialization=True,
):
    print(f"Saving learned embeddings to {save_path}")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(
            learned_embeds_dict, save_path, metadata={"format": "pt"}
        )
    else:
        torch.save(learned_embeds_dict, save_path)


def run_inference(prompt, pipeline, save_path):

    generator = torch.Generator(device="cpu").manual_seed(0)

    with torch.no_grad():
        print("### Running inference on the prompt: ", prompt)

        inf_img = pipeline(prompt, generator=generator).images[0]
        
        inf_img.save(save_path)

        return inf_img
