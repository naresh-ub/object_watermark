from watermark_args import parse_args
from utils.diffusers_imports import *
from utils.save_utils import get_psnr, save_comparison_image_3, save_progress
from utils.watermark_utils import iterative_denoising, iterative_denoising_for_NTI, unnormalize_vqgan

## import attacks for validation
from utils.watermark_utils import center_crop, resize, rotate, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, adjust_gamma, adjust_sharpness, jpeg_compress

## For attention maps
from utils.attn_map_utils import AttentionStore, show_cross_attention
from utils.ptp_utils import view_images, register_attention_control
from utils.cschedulers import customDDPMScheduler

## For logging
from accelerate.logging import get_logger
logger = get_logger(__name__)
    
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

    noise_scheduler = customDDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    return model, tokenizer, text_encoder, vae, unet, noise_scheduler
    
def add_token_for_TI(args, tokenizer, text_encoder, safetensors_checkpoint_path = None):
    
    from safetensors.torch import load_file
    
    # Add placeholder token
    placeholder_tokens = [args.placeholder_token]
    additional_tokens = [f"{args.placeholder_token}_{i}" for i in range(1, args.num_vectors)]
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(f"The tokenizer already contains the token {args.placeholder_token}.")

    # Convert tokens to ids and initialize embeddings
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")
    
    print("token_ids", token_ids)
    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    
    print("placeholder token ids", placeholder_token_ids)
    text_encoder.resize_token_embeddings(len(tokenizer))

    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
            
    embedding_tensor = None

    if safetensors_checkpoint_path is not None:
        # 4. Load the pretrained embedding from the safetensors file
        safetensor_dict = load_file(safetensors_checkpoint_path)

        # Assuming the safetensor dict has only one key corresponding to the token name, e.g., '<cat-toy>'
        if args.placeholder_token not in safetensor_dict:
            raise ValueError(f"The safetensors file does not contain an embedding for the token '{args.placeholder_token}'.")

        embedding_tensor = safetensor_dict[args.placeholder_token]
        
        print("From the function", embedding_tensor.shape)

        # 5. Replace the placeholder token embedding with the one from safetensors
        # Check if the embedding dimensions match
        expected_emb_dim = token_embeds.shape[-1]
        if embedding_tensor.shape[-1] != expected_emb_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {expected_emb_dim} but found {embedding_tensor.shape[-1]}"
            )

        # Update the embedding in the text encoder's input embeddings
        with torch.no_grad():
            token_embeds[placeholder_token_ids] = embedding_tensor.clone()
            
    return tokenizer, text_encoder, placeholder_token_ids, embedding_tensor

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # load models here
    
    model, tokenizer, text_encoder, vae, unet, noise_scheduler = get_stable_diff_models(args)
    
    tokenizer, text_encoder, placeholder_token_ids, embedding_tensor = add_token_for_TI(args, tokenizer, text_encoder)
    
    print("Starting embedding tensor", embedding_tensor)
    
    # Freeze parameters in VAE and U-Net
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Setting everything in Text Encoder to false other than token embeddings
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.train() # for gradient checkpointing only. taken from diffusers textual_inversion.py
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoader creation
    train_dataset = TextualInversionDataset(
        data_root=args.train_dir,
        tokenizer=tokenizer,
        placeholder_token=" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)),
        img_transforms=vqgan_transform_512,
        captions_file=args.train_captions,
    )
    
    valid_dataset = TextualInversionDataset(
        data_root=args.val_dir,
        tokenizer=tokenizer,
        placeholder_token=" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)),
        img_transforms=vqgan_transform_512,
        captions_file=args.val_captions,
    )
    
    # take random 10 images from datasets
    train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset))[:1])
    # valid_dataset = torch.utils.data.Subset(valid_dataset, torch.randperm(len(valid_dataset))[:10])
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Message decoder and loss setup
    whiten_save_path = f"assets/whitened_decoder_{args.num_bits}.pth"
    msg_decoder = return_msg_decoder(args, accelerator, whiten_save_path)
    loss_w, loss_i = return_losses(args, accelerator)

    key_str = "1110"
    key_tensor = torch.tensor([int(char) for char in key_str], dtype=torch.float32, device=accelerator.device).unsqueeze(0)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes, num_cycles=args.lr_num_cycles
    ) # this is not used in current train script

    text_encoder.train()
    text_encoder_gt = deepcopy(text_encoder)
    text_encoder_gt.eval()
    text_encoder, text_encoder_gt, optimizer, train_dataloader, valid_dataloader, lr_scheduler, msg_decoder = accelerator.prepare(
        text_encoder, text_encoder_gt, optimizer, train_dataloader, valid_dataloader, lr_scheduler, msg_decoder
    )

    # Move VAE and UNet to the correct device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.report_to == "wandb" and accelerator.is_main_process:
        accelerator.init_trackers("Oct_23_TI", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps", disable=not accelerator.is_local_main_process)

    # Keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save (Important)
    if args.resume_from_checkpoint:
        # Get the most recent checkpoint
        dirs = os.listdir(args.checkpoint_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.eval()
        text_encoder.train()
        msg_decoder.eval()
        vae.eval()
        
        train_loss = 0.0
        latent_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(text_encoder):
                    
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                
                latents = latents * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(args.t_start, args.t_end, (bsz,), device=latents.device).long()
                
                noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)
                
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                encoder_hidden_states_f = text_encoder(batch["input_ids_1"])[0].to(dtype=weight_dtype)

                noisy_latent_f = noisy_latent.clone() 

                for i in range(args.t_start): # manually set to t_start steps
                    # Calculate model predictions for the current noisy latent
                    model_pred = unet(noisy_latent, timesteps - i, encoder_hidden_states).sample
                    model_pred_f = unet(noisy_latent_f, timesteps - i, encoder_hidden_states_f).sample
                    
                    # Use noise_scheduler.step to get the next step's denoised latent
                    noisy_latent = noise_scheduler.step(model_pred, timesteps - i, noisy_latent).prev_sample
                    noisy_latent_f = noise_scheduler.step(model_pred_f, timesteps - i, noisy_latent_f).prev_sample
                    
                    del model_pred, model_pred_f
                    torch.cuda.empty_cache()

                imgs_w = vae.decode(noisy_latent / vae.config.scaling_factor).sample
                imgs_d0 = vae.decode(noisy_latent_f / vae.config.scaling_factor).sample
                
                # calculate train PSNR
                train_psnr = get_psnr(imgs_w, imgs_d0)

                decoded = msg_decoder(vqgan_to_imnet_512(imgs_w))
                keys = key_tensor.repeat(latents.shape[0], 1)

                lossw = loss_w(decoded, keys)
                latent_loss = F.mse_loss(noisy_latent, noisy_latent_f)
                # diff_loss = F.mse_loss(model_pred, model_pred_f)

                loss = lossw + 10 * latent_loss
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                train_bit_accs = (torch.sum(~torch.logical_xor(decoded > 0, keys > 0), dim=-1) / decoded.shape[-1]).mean()
                
                # Every 10 steps, plot and save comparison images
                if step % 10 == 0:
                    output_path = os.path.join(args.output_dir, "train_comparison_image.png")
                    plot_comparison_images(imgs_w, imgs_d0, train_bit_accs, train_psnr, output_path)
                
                del latents, noise, noisy_latent, encoder_hidden_states, encoder_hidden_states_f, noisy_latent_f
                
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

                # Call CUDA garbage collector
                torch.cuda.empty_cache()
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % 120 == 0:
                        
                        weight_name = (
                            f"checkpoint-{global_step}.bin"
                            if args.no_safe_serialization
                            else f"checkpoint-{global_step}.safetensors"
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
                        
                        print("Ground Truth", text_encoder.get_input_embeddings().weight.data[-1][:5])
                        
            logs = {"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "bit_acc": train_bit_accs.item(), "latent_loss": latent_loss.item(), "loss_w": lossw.item(), "train_psnr": train_psnr}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if global_step % 120 == 0:
                attacks = {
                    'none': lambda x: x,  # No attack
                    'center_crop': lambda x: center_crop(x, 0.5),  # Center crop with scale of 0.5
                    'resize': lambda x: resize(x, 0.5),  # Resize with scale of 0.5
                    'rotate_25': lambda x: rotate(x, 25),  # Rotate by 25 degrees
                    'adjust_brightness_1.5': lambda x: adjust_brightness(x, 1.5),  # Adjust brightness by 1.5
                    'adjust_contrast_1.5': lambda x: adjust_contrast(x, 1.5),  # Adjust contrast by 1.5
                    'adjust_saturation_1.5': lambda x: adjust_saturation(x, 1.5),  # Adjust saturation by 1.5
                    'adjust_hue_0.1': lambda x: adjust_hue(x, 0.1),  # Adjust hue by 0.1
                    'adjust_gamma_0.8': lambda x: adjust_gamma(x, 0.8),  # Adjust gamma by 0.8
                    'adjust_sharpness_1.5': lambda x: adjust_sharpness(x, 1.5),  # Adjust sharpness by 1.5
                    'jpeg_compress_80': lambda x: jpeg_compress(x, 80)  # JPEG compression with quality 80
                }
                
                # model, tokenizer, val_text_encoder, vae, unet, noise_scheduler = get_stable_diff_models(args)
        
                # TI_path = f"{args.output_dir}/checkpoint-{global_step}.safetensors"
                
                # print("loading TI from", TI_path)
                
                # tokenizer, val_text_encoder, val_placeholder_token_ids, embedding_tensor = add_token_for_TI(args, tokenizer, val_text_encoder, safetensors_checkpoint_path=TI_path)
            
                val_text_encoder = accelerator.unwrap_model(text_encoder)
                val_text_encoder.to(accelerator.device)
                vae.to(accelerator.device)
                unet.to(accelerator.device)
                
                unet.eval()
                val_text_encoder.eval()
                vae.eval()
                msg_decoder.eval()
                
                valid_loss = 0.0
                valid_latent_loss = 0.0
                total_bit_acc = 0.0
                total_psnr = 0.0

                # Create a dictionary to store bit accuracies for each attack
                attack_bit_accs = {attack: 0.0 for attack in attacks.keys()}

                for step, batch in enumerate(train_dataloader):
                    with torch.no_grad():
                        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                        latents = latents * vae.config.scaling_factor

                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        timesteps = torch.randint(args.t_start, args.t_end, (bsz,), device=latents.device).long()

                        noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)
                        encoder_hidden_states = val_text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                        model_pred = unet(noisy_latent, timesteps, encoder_hidden_states).sample

                        encoder_hidden_states_f = val_text_encoder(batch["input_ids_1"])[0].to(dtype=weight_dtype)
                        model_pred_f = unet(noisy_latent, timesteps, encoder_hidden_states_f).sample

                        denoised_latent = noise_scheduler.subtract_noise(noisy_latent, model_pred, timesteps)
                        denoised_latent_f = noise_scheduler.subtract_noise(noisy_latent, model_pred_f, timesteps)

                        imgs_w = vae.decode(denoised_latent / vae.config.scaling_factor).sample
                        imgs_d0 = vae.decode(denoised_latent_f / vae.config.scaling_factor).sample

                        # Calculate validation PSNR for the current batch
                        valid_psnr = get_psnr(imgs_w, imgs_d0)

                        # Iterate over each attack, apply it to imgs_w, and calculate bit accuracy
                        for attack_name, attack_fn in attacks.items():
                            
                            # Apply the attack to imgs_w
                            attacked_imgs_w = attack_fn(imgs_w)

                            # Run the attacked images through the message decoder
                            decoded = msg_decoder(vqgan_to_imnet_512(attacked_imgs_w))
                            keys = key_tensor.repeat(latents.shape[0], 1)

                            lossw = loss_w(decoded, keys)
                            latent_loss = F.mse_loss(denoised_latent, denoised_latent_f)
                            diff_loss = F.mse_loss(model_pred, model_pred_f)

                            loss = lossw + 10 * latent_loss
                            avg_valid_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()

                            valid_bit_accs = (torch.sum(~torch.logical_xor(decoded > 0, keys > 0), dim=-1) / decoded.shape[-1]).mean()
                            attack_bit_accs[attack_name] += valid_bit_accs.item() / len(train_dataloader)

                        del latents, noise, noisy_latent, encoder_hidden_states, model_pred, encoder_hidden_states_f, model_pred_f, denoised_latent, denoised_latent_f
                        torch.cuda.empty_cache()

                # Log the bit accuracies for each attack
                for attack_name, bit_acc in attack_bit_accs.items():
                    logger.info(f"Attack: {attack_name}, Bit Accuracy: {bit_acc:.4f}")
                    logs = {f"valid_bit_acc_{attack_name}": bit_acc}
                    accelerator.log(logs, step=global_step)
                
            # progress_bar.set_postfix(valid_loss=valid_loss, valid_bit_acc=avg_bit_acc, valid_psnr=avg_psnr)

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        weight_name = (
            f"checkpoint-{global_step}.bin"
            if args.no_safe_serialization
            else f"checkpoint-{global_step}.safetensors"
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

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)