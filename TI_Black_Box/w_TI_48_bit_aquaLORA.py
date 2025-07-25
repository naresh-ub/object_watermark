from watermark_args import parse_args
from utils.diffusers_imports import *
from utils.save_utils import get_psnr, save_comparison_image_3, save_progress
from utils.watermark_utils import iterative_denoising, iterative_denoising_for_NTI
from utils.data import TextualInversionDatasetOld

## import attacks for validation
from utils.watermark_utils import center_crop, resize, rotate, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, adjust_gamma, adjust_sharpness, jpeg_compress

## AquaLORA utils
from utils.aqualora_models import SecretEncoder, SecretDecoder, MapperNet

def tensor_to_npimg(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # Convert from CHW to HWC
    img = (img + 1.0) * 127.5  # Convert back from [-1, 1] to [0, 255]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def plot_comparison_images(imgs_w, imgs_d0, bit_acc, psnr, output_path):
    """
    Plots comparison images between imgs_w, imgs_d0, and their difference, 
    then saves them with a title including bit accuracy and PSNR.

    Args:
        imgs_w (Tensor): Generated images (denoised).
        imgs_d0 (Tensor): Target images (ground truth).
        bit_acc (float): Bit accuracy for the current batch.
        psnr (float): PSNR value for the current batch.
        output_path (str): Path to save the image comparison plot.
    """
    # Move images to CPU and convert to numpy arrays
    imgs_w = imgs_w.detach().cpu().numpy().transpose(0, 2, 3, 1)
    imgs_d0 = imgs_d0.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # Calculate absolute difference (diff_image)
    diff_image = abs(imgs_w - imgs_d0)
    
    # Plot first image from the batch (batch index 0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Preprocessing for matplotlib (images should be in [0, 1] range)
    img_w = (imgs_w[0] - imgs_w[0].min()) / (imgs_w[0].max() - imgs_w[0].min())
    img_d0 = (imgs_d0[0] - imgs_d0[0].min()) / (imgs_d0[0].max() - imgs_d0[0].min())
    img_diff = (diff_image[0] - diff_image[0].min()) / (diff_image[0].max() - diff_image[0].min())

    # Display the images
    axes[0].imshow(img_w)
    axes[0].set_title("Generated Image (imgs_w)")
    axes[0].axis("off")

    axes[1].imshow(img_d0)
    axes[1].set_title("Target Image (imgs_d0)")
    axes[1].axis("off")

    axes[2].imshow(img_diff)
    axes[2].set_title("Difference (|imgs_w - imgs_d0|)")
    axes[2].axis("off")
    
    bit_acc_value = bit_acc.item() if isinstance(bit_acc, torch.Tensor) else bit_acc
    
    psnr_value = psnr.item() if isinstance(psnr, torch.Tensor) else psnr
    

    # Set the main title with bit accuracy and PSNR
    fig.suptitle(f"Bit Acc: {bit_acc_value:.4f}, PSNR: {psnr_value:.2f}", fontsize=16)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def get_stable_diff_models(args):
    
    # Model initialization
    model = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer = model.tokenizer
    text_encoder = model.text_encoder
    vae = model.vae
    unet = model.unet

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    return model, tokenizer, text_encoder, vae, unet, noise_scheduler
    
def add_token_for_TI(tokenizer, text_encoder, safetensors_checkpoint_path=None, token="<cat-toy>", init_token = "toy"):
    from safetensors.torch import load_file

    # Add placeholder token
    placeholder_tokens = [token]
    additional_tokens = [f"{token}_{i}" for i in range(1, 1)]
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != 1:
        raise ValueError(f"The tokenizer already contains the token {token}.")

    token_ids = tokenizer.encode(init_token, add_special_tokens=False)

    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    text_encoder.resize_token_embeddings(len(tokenizer))

    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    embedding_tensor = None

    if safetensors_checkpoint_path is not None:
        safetensor_dict = load_file(safetensors_checkpoint_path)

        if token not in safetensor_dict:
            raise ValueError(f"The safetensors file does not contain an embedding for the token '{token}'.")

        embedding_tensor = safetensor_dict[token]

        expected_emb_dim = token_embeds.shape[-1]
        if embedding_tensor.shape[-1] != expected_emb_dim:
            padded_embedding = torch.zeros(expected_emb_dim, device=embedding_tensor.device)
            padded_embedding[: embedding_tensor.shape[-1]] = embedding_tensor
            embedding_tensor = padded_embedding

        with torch.no_grad():
            token_embeds[placeholder_token_ids] = embedding_tensor.clone()

    return tokenizer, text_encoder, placeholder_token_ids, embedding_tensor

attacks = {
    'none': lambda x: x,
    'crop_0.1': lambda x: center_crop(x, 0.1),
    'crop_0.5': lambda x: center_crop(x, 0.5),
    'rotate_25': lambda x: rotate(x, 25),
    'rotate_90': lambda x: rotate(x, 90),
    'resize_3': lambda x: resize(x, 0.3),
    'resize_7': lambda x: resize(x, 0.7),
    'adjust_brightness_1.5': lambda x: adjust_brightness(x, 1.5),
    'adjust_brightness_2.0': lambda x: adjust_brightness(x, 2.0),
    'adjust_contrast_1.5': lambda x: adjust_contrast(x, 1.5),
    'adjust_saturation_1.5': lambda x: adjust_saturation(x, 1.5),
    'adjust_hue_0.1': lambda x: adjust_hue(x, 0.1),
    'adjust_gamma_0.8': lambda x: adjust_gamma(x, 0.8),
    'adjust_sharpness_1.5': lambda x: adjust_sharpness(x, 1.5),
    'jpeg_compress_80': lambda x: jpeg_compress(x, 80),
    'jpeg_compress_50': lambda x: jpeg_compress(x, 50),
}

def main():
    args = parse_args()

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, text_encoder, vae, unet, noise_scheduler = get_stable_diff_models(args)
    
    tokenizer, text_encoder, placeholder_token_ids, embedding_tensor = add_token_for_TI(tokenizer, text_encoder, token = args.placeholder_token, init_token=args.initializer_token)
    
    print("Starting embedding tensor", embedding_tensor)

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        pass
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        # unet.train()
        # text_encoder.gradient_checkpointing_enable()
        # unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # if args.scale_lr:
    #     args.learning_rate = 0.002
        
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)), # already resized to 512 x 512
            transforms.RandomHorizontalFlip() if False else transforms.Lambda(lambda x: x), # adding random flip manually
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_dir,
        tokenizer=tokenizer,
        placeholder_token=(
            " ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))
        ),
        img_transforms=train_transforms,
        captions_file=args.train_captions,
    )

    valid_dataset = TextualInversionDataset(
        data_root=args.val_dir,
        tokenizer=tokenizer,
        placeholder_token=(
            " ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))
        ),
        img_transforms=train_transforms,
        captions_file=args.val_captions,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    ## AquaLORA msgdecoder setup
    key_tensor = torch.randint(0, 2, (args.train_batch_size, args.aqualora_bits)).float().to(accelerator.device) # 48 bit key
    key_str = "".join([str(int(ii)) for ii in key_tensor.tolist()[0]])
    print("Key tensor", key_str)
    
    pretrain_dict = torch.load(args.aqualora_pretrain_path)
    sec_encoder = SecretEncoder(secret_len=args.aqualora_bits, resolution=64)
    sec_encoder.load_state_dict(pretrain_dict['sec_encoder'])
    
    msg_decoder = SecretDecoder(output_size=args.aqualora_bits)
    msg_decoder.load_state_dict(pretrain_dict['sec_decoder'])

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    text_encoder.train()
    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    sec_encoder.to(accelerator.device, dtype=weight_dtype)
    msg_decoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # if args.report_to == "wandb":
    if accelerator.is_main_process:
        accelerator.init_trackers("Oct_30_plug_and_play", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    global_step = 0
    first_epoch = 0
    gen_count = 0
    initial_global_step = 0
    step_count = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # keep original embeddings as reference
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight.data.clone()
    )

    # first_batch = next(iter(train_dataloader))

    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        msg_decoder.eval()
        sec_encoder.eval()
        unet.eval()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):

                # Convert images to latent space
                latents = (
                    vae.encode(batch["pixel_values"].to(dtype=weight_dtype))
                    .latent_dist.sample()
                    .detach()
                )
                latents = latents * vae.config.scaling_factor

                keys = key_tensor.repeat(latents.shape[0], 1)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    args.t_start,
                    args.t_end,
                    (bsz,),
                    device=latents.device,
                )
                
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = []
                
                for i in range(timesteps):
                    noisy_latents.append(noise_scheduler.add_noise(latents, noise, torch.tensor([i], device=latents.device)))

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                encoder_hidden_states_f = text_encoder(batch["input_ids_1"])[0].to(dtype=weight_dtype)
                null_embeds = text_encoder(batch["input_null"])[0].to(dtype=weight_dtype)

                # Perform iterative denoising
                if args.denoise_steps is None:
                    denoise_steps = int(timesteps.cpu().detach().numpy())
                else:
                    denoise_steps = args.denoise_steps

                denoised_last_latent, _ , denoised_latents = iterative_denoising_for_NTI(
                    timesteps,
                    noisy_latents[-1],
                    noise_scheduler,
                    unet,
                    encoder_hidden_states,
                    encoder_hidden_states_f,
                    null_embeds,
                    denoise_timesteps=args.denoise_steps,
                    mode = "watermark"
                )
                
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    denoised_last_latent_f, _ , denoised_latents_f = iterative_denoising_for_NTI(
                        timesteps,
                        noisy_latents[-1],
                        noise_scheduler,
                        unet,
                        _,
                        encoder_hidden_states_f,
                        null_embeds,
                        denoise_timesteps=args.denoise_steps,
                        mode = "no_watermark"
                    )
                
                torch.cuda.empty_cache()

                noisy_latents_order = noisy_latents[::-1]

                NT_loss = 0.0
                for noisy_latent, denoised_latent in zip(noisy_latents_order, denoised_latents):
                    NT_loss += F.mse_loss(denoised_latent, noisy_latent, reduction = "mean")

                # Optionally, you can take the mean MSE across all the timesteps
                NT_loss = NT_loss / len(denoised_latents)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                imgs_w = vae.decode(denoised_last_latent / vae.config.scaling_factor)[0]
                imgs_d0 = vae.decode(denoised_last_latent_f / vae.config.scaling_factor)[0]
                
                imgs_input = batch["pixel_values"].to(dtype=weight_dtype)

                decoded = msg_decoder(imgs_w.float())
                
                # Create labels tensor
                key_tensor_int = key_tensor.to(torch.int64)
                labels = F.one_hot(key_tensor_int, num_classes=2).float()
                
                # change to BCE loss
                msgloss = F.binary_cross_entropy_with_logits(decoded, labels.cuda())
                
                lossw = msgloss

                loss = 100 * lossw + 100 * NT_loss

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                del noisy_latents, denoised_latents, noisy_latents_order, denoised_last_latent, denoised_last_latent_f, encoder_hidden_states, encoder_hidden_states_f
                
                decoded_key = torch.argmax(decoded, dim=2)

                bit_accs = 1 - torch.abs(decoded_key - key_tensor).sum().float() / (48)

                img_metrics = {
                    "psnr": get_psnr(imgs_d0, imgs_w).cpu().detach().numpy()[0],
                    "timesteps": float(timesteps.cpu().detach().numpy()[0]),
                    "bit_acc": bit_accs.mean().item(),
                    "step": step_count,
                    # "ssim": get_ssim(imgs_d0, imgs_w),
                }

                # Logging for training
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "global_step": global_step,
                    "Bit_acc": bit_accs.mean().item(),  # Training bit accuracy
                    "loss_w": 100 * lossw.detach().item(),
                    # "loss_i": lossi.detach().item(),
                    "psnr": get_psnr(imgs_d0, imgs_w).cpu().detach().numpy()[0],
                    "NT_loss": 100 * NT_loss.detach().item(),
                }

                imgs_w = imgs_w.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                imgs_input = (
                    imgs_input.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                )
                imgs_d0 = (
                    imgs_d0.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                )

                save_comparison_image_3(
                    imgs_d0,
                    imgs_w,
                    args.output_dir,
                    epoch,
                    global_step,
                    label="train_images",
                    img_metrics=img_metrics
                )

                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[
                    min(placeholder_token_ids) : max(placeholder_token_ids) + 1
                ] = False

                with torch.no_grad():
                    accelerator.unwrap_model(
                        text_encoder
                    ).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[
                        index_no_updates
                    ]

                if accelerator.sync_gradients:
                    
                    if accelerator.is_main_process:
                        
                        # Validation every 400 steps after 2000 steps
                        if step_count % 1000 == 0 and step_count > 20000:
    
                            text_encoder_eval = accelerator.unwrap_model(text_encoder)
                            avg_val_bit_acc = {attack_name: 0.0 for attack_name in attacks.keys()}
                            avg_psnr_value = 0.0
                            total_samples = {attack_name: 0 for attack_name in attacks.keys()}
                            total_psnr_samples = 0
                            
                            total_val_loss = 0.0
                            total_loss_samples = 0

                            with torch.no_grad():
                                for val_step, val_batch in enumerate(valid_dataloader):

                                    # Convert validation images to latent space
                                    val_latents = (
                                        vae.encode(val_batch["pixel_values"].to(dtype=weight_dtype))
                                        .latent_dist.sample()
                                        .detach()
                                    )

                                    # print("Validation prompt", val_batch['text'])

                                    val_latents = val_latents * vae.config.scaling_factor

                                    # Generate random key for watermarking
                                    val_keys = key_tensor.repeat(val_latents.shape[0], 1)

                                    # Sample noise and timesteps for validation
                                    val_noise = torch.randn_like(val_latents)
                                    
                                    val_bsz = val_latents.shape[0]
                                    
                                    val_timesteps = torch.randint(
                                        args.t_start,
                                        args.t_end,
                                        (val_bsz,),
                                        device=val_latents.device,
                                    )
                                    val_timesteps = val_timesteps.long()

                                    val_noisy_latents = []
                                    for i in range(val_timesteps):
                                        val_noisy_latents.append(
                                            noise_scheduler.add_noise(val_latents, val_noise, torch.tensor([i], device=val_latents.device))
                                        )

                                    # Get the text embedding for validation
                                    val_encoder_hidden_states = text_encoder_eval(val_batch["input_ids"])[0].to(dtype=weight_dtype)
                                    val_encoder_hidden_states_f = text_encoder_eval(val_batch["input_ids_1"])[0].to(dtype=weight_dtype)
                                    val_null_embeds = text_encoder_eval(val_batch["input_null"])[0].to(dtype=weight_dtype)

                                    # Perform iterative denoising
                                    val_denoise_steps = int(val_timesteps.cpu().detach().numpy()) if args.denoise_steps is None else args.denoise_steps

                                    val_denoised_latents, _, _ = iterative_denoising_for_NTI(
                                        val_timesteps,
                                        val_noisy_latents[-1],
                                        noise_scheduler,
                                        unet,
                                        val_encoder_hidden_states,
                                        _,
                                        val_null_embeds,
                                        denoise_timesteps=val_denoise_steps,
                                        mode = "watermark"
                                    )
                                    
                                    val_denoised_latents_f, _, _ = iterative_denoising_for_NTI(
                                        val_timesteps,
                                        val_noisy_latents[-1],
                                        noise_scheduler,
                                        unet,
                                        _,
                                        val_encoder_hidden_states_f,
                                        val_null_embeds,
                                        denoise_timesteps=val_denoise_steps,
                                        mode = "no_watermark"
                                    )

                                    # Decode and perform attack-based evaluation
                                    for attack_name, attack_fn in attacks.items():
                                        # Apply attack and decode images
                                        val_imgs_w = vae.decode(val_denoised_latents / vae.config.scaling_factor)[0]
                                        # Decode the 'no watermark' images as reference
                                        val_imgs_d0 = vae.decode(val_denoised_latents_f / vae.config.scaling_factor)[0]
                                        
                                        psnr_value = get_psnr(val_imgs_d0, val_imgs_w).cpu().detach().numpy()[0]
                                        avg_psnr_value += psnr_value * val_bsz
                                        total_psnr_samples += val_bsz
                                        
                                        val_imgs_attacked = attack_fn(val_imgs_w)

                                        # Bit accuracy calculation for attacked validation
                                        val_decoded = msg_decoder(val_imgs_attacked.float())
                                        
                                        val_key_tensor_int = val_keys.to(torch.int64)
                                        val_labels = F.one_hot(val_key_tensor_int, num_classes=2).float()
                                        # change to BCE loss
                                        val_msgloss = F.binary_cross_entropy_with_logits(val_decoded, val_labels.cuda())
                                        
                                        val_loss_w = val_msgloss
                                        
                                        total_val_loss += val_loss_w.item() * val_bsz
                                        total_loss_samples += val_bsz

                                        val_decoded = torch.argmax(val_decoded, dim=2)
                                        
                                        val_bit_acc = 1 - torch.abs(val_decoded - val_keys).sum().float() / (48)

                                        avg_val_bit_acc[attack_name] += torch.sum(val_bit_acc).item() * val_bsz
                                        
                                        total_samples[attack_name] += val_bsz

                                        val_img_metrics = {
                                            "psnr": psnr_value,
                                            "timesteps": float(val_timesteps.cpu().detach().numpy()[0]),
                                            "bit_acc": val_bit_acc.mean().item(),
                                            "step": step_count,
                                        }

                                        val_imgs_w = val_imgs_w.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                                        val_imgs_d0 = val_imgs_d0.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                                        
                                        # val_save_path = os.path.join(args.output_dir, f"val_images_{attack_name}_{step_count}")

                                        # if not os.path.exists(val_save_path):
                                        #     os.makedirs(val_save_path)

                                        # save_comparison_image_3(
                                        #     val_imgs_d0,
                                        #     val_imgs_w,
                                        #     val_save_path,
                                        #     epoch,
                                        #     total_samples[attack_name],
                                        #     label=f"valid_images_{attack_name}",
                                        #     img_metrics=val_img_metrics
                                        # )

                                        # print(f"Saved validation image to {val_save_path} for attack {attack_name}")

                            # Average validation bit accuracy for each attack
                            avg_val_bit_acc = {attack_name: avg_val_bit_acc[attack_name] / total_samples[attack_name] if total_samples[attack_name] > 0 else 0.0 for attack_name in attacks.keys()}
                            
                            avg_psnr_value = avg_psnr_value / total_psnr_samples if total_psnr_samples > 0 else 0.0
                            
                            avg_valid_loss = total_val_loss / total_loss_samples if total_loss_samples > 0 else 0.0

                            # Logging for validation with different attacks
                            logs.update({
                                f"valid_bit_acc_{attack_name}": avg_val_bit_acc[attack_name] for attack_name in attacks.keys()
                            })
                            logs["valid_avg_psnr"] = avg_psnr_value
                            logs["valid_loss"] = avg_valid_loss
                            
                            # print(">>>>>>", logs)
                            
                            # return

                    # Save the learned embeddings when bit accuracy is high enough
                    if bit_accs.mean().item() >= 0.95:
                        weight_name = (
                            f"learned_embeds-steps-{step_count}.bin"
                            if args.no_safe_serialization
                            else f"learned_embeds-steps-{step_count}_{bit_accs.item():.4f}.safetensors"
                        )
                        save_path = os.path.join(args.output_dir, weight_name)
                        save_progress(
                            text_encoder,
                            placeholder_token_ids,
                            accelerator,
                            args,
                            save_path,
                            safe_serialization=not args.no_safe_serialization,
                        )

                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=step_count)

                    step_count += 1
                    progress_bar.update(1)

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        pass

            if step_count >= args.max_train_steps:
                break
            
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pass

    accelerator.end_training()


if __name__ == "__main__":
    main()