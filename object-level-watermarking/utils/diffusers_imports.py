# Common imports
import argparse
import json
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from torch.utils.data import Dataset, Subset

import accelerate
import cv2

## Diffusers imports
import diffusers
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

if is_wandb_available():
    import wandb

logger = get_logger(__name__)
LOW_RESOURCE = False

import os

import numpy as np
from PIL import Image

torch.autograd.set_detect_anomaly(True)

# Project specific imports
import sys

sys.path.append("/home/csgrad/devulapa/w_AIGC")

from copy import deepcopy

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from loss.loss_provider import LossProvider

## Textual Inversion utils
from utils.data import ImageCaptionDataset, TextualInversionDataset, HuggingFaceTextualInversionDataset

## Stable Signature utils
from utils.hidden_msg_decoder import ConvBNRelu, HiddenDecoder, HiddenEncoder
from utils.save_utils import decode_images_and_plot, run_inference, save_progress
from utils.watermark_utils import (
    get_psnr,
    get_ssim,
    iterative_denoising,
    msg2str,
    return_losses,
    return_msg_decoder,
    str2msg,
    vqgan_to_imnet_512,
    vqgan_transform_512,
)

# from w_AIGC.utils.watermark_args_parse import parse_args
# from utils.txt2img_train_args import parse_args
# from utils.attn_map_utils import AttentionStore, show_cross_attention
# from utils.ptp_utils import view_images, register_attention_control
