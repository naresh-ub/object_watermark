import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(  # this is set to false mostly always
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(  # important
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--hidden_whiten_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default=None,
        help="A token to use as initializer word.",
    )
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="object",
        help="Choose between 'object' and 'style'",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="How many times to repeat the training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--ti_resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )

    ## Stable Signature Args

    parser.add_argument(
        "--msg_decoder_path",
        type=str,
        default=None,
        help="Path to the message decoder checkpoint.",
    )

    parser.add_argument(
        "--num_bits",
        type=int,
        default=None,
        help="Number of bits to use for the message.",
    )

    parser.add_argument(
        "--redundancy",
        type=int,
        default=1,
        help="Redundancy to use for the message.",
    )

    parser.add_argument(
        "--decoder_depth",
        type=int,
        default=8,
        help="Depth of the decoder.",
    )

    parser.add_argument(
        "--decoder_channels",
        type=int,
        default=64,
        help="Channels of the decoder.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Watermark or None.",
    )

    parser.add_argument(
        "--loss_i",
        type=str,
        default="watson-vgg",
        help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.",
    )

    parser.add_argument(
        "--loss_w",
        type=str,
        default="bce",
        help="Type of loss for the watermark loss. Can be mse or bce",
    )

    parser.add_argument(
        "--lambda_i",
        type=float,
        default=0.2,
        help="Weight of the image loss in the total loss",
    )

    parser.add_argument(
        "--lambda_w",
        type=float,
        default=1.0,
        help="Weight of the watermark loss in the total loss",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="A seed for reproducible training.",
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="Path to the whitening data.",
    )

    parser.add_argument(
        "--train_captions",
        type=str,
        default=None,
        help="Path to the captions file.",
    )

    parser.add_argument(
        "--hidden_resolution",
        type=int,
        default=None,
        help="Resolution for whitening hidden.",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="The project name for wandb.",
    )

    parser.add_argument(
        "--t_start",
        type=int,
        default=0,
        help="Start for forward diffusion.",
    )

    parser.add_argument(
        "--t_end",
        type=int,
        default=1000,
        help="End for forward diffusion.",
    )

    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )

    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--add_diffusion",
        type=str,
        default="False",
        help="Whether to use forward and reverse diffusion.",
    )

    parser.add_argument(
        "--inf_step",
        type=int,
        default=10,
        help="When to start inference.",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help="When to end inference.",
    )

    parser.add_argument(
        "--TI_lambda",
        type=float,
        default=None,
        help="When to end inference.",
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The run name for wandb.",
    )

    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Denoising steps.",
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Validation dir",
    )

    parser.add_argument(
        "--val_captions",
        type=str,
        default=None,
        help="Validation captions",
    )

    parser.add_argument(
        "--lambda_NT",
        type=float,
        default=None,
        help="NT Loss Lambda",
    )
    
    parser.add_argument(
        "--aqualora_pretrain_path",
        type=str,
        default=None,
        help="Aqualora Pretrain Path",
    )
    
    parser.add_argument(
        "--aqualora_bits",
        type=int,
        default=48,
        help="Aqualora Bits",
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint dir",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args
