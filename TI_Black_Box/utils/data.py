import json
import os
import random

import numpy as np
import PIL
import torch
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, caption_file, tokenizer, transform=None):
        """
        Args:
            image_folder (str): Path to the folder containing images.
            caption_file (str): Path to the JSON file containing captions.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = tokenizer

        print("Loading captions from", caption_file)

        # Load JSON file and get image filenames
        with open(caption_file, "r") as f:
            self.captions = json.load(f)

        # Create a list of image filenames
        self.image_filenames = os.listdir(self.image_folder)

        # Create a lookup dictionary for quick access to captions by image filename
        self.caption_lookup = {
            entry["image_id"]: entry["caption"] for entry in self.captions
        }

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        example = {}
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")

        # Lookup the caption for the image
        caption = self.caption_lookup[img_name]

        if self.transform:
            image = self.transform(image)

        example["pixel_values"] = image
        example["text"] = caption
        example["input_ids"] = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example['file_name'] = img_name

        # Add null text embeddings (unconditioned text) for classifier-free guidance
        null_caption = ""  # This represents the null/unconditioned text
        example["input_null"] = self.tokenizer(
            null_caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return example


## Textual Inversion Utils

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


# class TextualInversionDataset(Dataset):
#     """
#     Works for both watermarking and vanilla Textual Inversion
#     """

#     def __init__(
#         self,
#         data_root,
#         tokenizer,
#         learnable_property="object",  # [object, style]
#         size=512,
#         repeats=100,
#         interpolation="bicubic",
#         flip_p=0.5,
#         set="train",
#         placeholder_token="*",
#         center_crop=False,
#         img_transforms=None,
#         captions_file=None,
#     ):
#         self.data_root = data_root
#         self.tokenizer = tokenizer
#         self.learnable_property = learnable_property
#         self.size = size
#         self.placeholder_token = placeholder_token
#         self.center_crop = center_crop
#         self.flip_p = flip_p
#         self.img_transforms = img_transforms
#         self.captions_file = captions_file

#         self.image_paths = [
#             os.path.join(self.data_root, file_path)
#             for file_path in os.listdir(self.data_root)
#         ]
#         self.num_images = len(self.image_paths)
#         self._length = self.num_images

#         if set == "train":
#             self._length = self.num_images * repeats

#         self.interpolation = {
#             "linear": PIL_INTERPOLATION["linear"],
#             "bilinear": PIL_INTERPOLATION["bilinear"],
#             "bicubic": PIL_INTERPOLATION["bicubic"],
#             "lanczos": PIL_INTERPOLATION["lanczos"],
#         }[interpolation]

#         self.templates = (
#             imagenet_style_templates_small
#             if learnable_property == "style"
#             else imagenet_templates_small
#         )
#         self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

#         # Load captions from JSON file if provided
#         self.captions = {}
#         if self.captions_file:
#             with open(self.captions_file, "r") as f:
#                 captions_data = json.load(f)
#                 for item in captions_data:
#                     self.captions[item["image_id"]] = item["caption"]

#     def __len__(self):
#         return self._length

#     def __getitem__(self, i):
#         example = {}
#         image_path = self.image_paths[i % self.num_images]
#         image = Image.open(image_path)

#         if not image.mode == "RGB":
#             image = image.convert("RGB")

#         # Extract the image name
#         image_name = os.path.basename(image_path)
        
#         if image_name.endswith(".png"):
#             image_name = image_name.replace(".png", ".jpg")

#         # Use the caption from the JSON file if available, otherwise use a template
#         if self.captions_file is not None:
#             if image_name in self.captions:
#                 text = self.captions[image_name] + " " + self.placeholder_token
#                 text1 = self.captions[
#                     image_name
#                 ]  # does not contain the placeholder token
#             else:
#                 raise ValueError(f"Caption not found for image {image_name}")
#         else:
#             placeholder_string = self.placeholder_token
#             text = random.choice(self.templates).format(placeholder_string)
            
#             # Replace placeholder_string with "cat"
#             text1 = text.replace(placeholder_string, "cat toy")
        
#         example["input_ids"] = self.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.tokenizer.model_max_length,
#             return_tensors="pt",
#         ).input_ids[0]

#         example["input_ids_1"] = self.tokenizer(
#             text1,
#             padding="max_length",
#             truncation=True,
#             max_length=self.tokenizer.model_max_length,
#             return_tensors="pt",
#         ).input_ids[0]

#         null_caption = ""  # This represents the null/unconditioned text
#         example["input_null"] = self.tokenizer(
#             null_caption,
#             padding="max_length",
#             truncation=True,
#             max_length=self.tokenizer.model_max_length,
#             return_tensors="pt",
#         ).input_ids[0]

#         example["input_images"] = torch.from_numpy(np.array(image)).permute(
#             2, 0, 1
#         )  # [0, 255]
#         example["text"] = text
#         example["text1"] = text1

#         if self.img_transforms is not None:
#             image = self.img_transforms(image)
#             # print("img_transforms in dataset class range: ", image.min(), image.max())
#             # this prints [-1, 1] #checked
#             example["pixel_values"] = image

#         else:
#             img = np.array(image).astype(np.uint8)

#             if self.center_crop:
#                 crop = min(img.shape[0], img.shape[1])
#                 (
#                     h,
#                     w,
#                 ) = (
#                     img.shape[0],
#                     img.shape[1],
#                 )
#                 img = img[
#                     (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
#                 ]

#             image = Image.fromarray(img)

#             image = image.resize((self.size, self.size), resample=self.interpolation)

#             image = self.flip_transform(image)
#             image = np.array(image).astype(np.uint8)
#             image = (image / 127.5 - 1.0).astype(np.float32)

#             example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

#         return example

class TextualInversionDataset(Dataset):
    """
    A dataset class that reads from a folder of images 
    and a captions.json file, and applies placeholder tokens in text prompts.
    """

    def __init__(
        self,
        data_root,
        tokenizer,
        placeholder_token="*",
        img_transforms=None,
        captions_file=None,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.img_transforms = img_transforms
        self.captions_file = captions_file

        # Load image paths
        self.image_paths = [
            os.path.join(self.data_root, file_name)
            for file_name in os.listdir(self.data_root)
        ]
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        # Load captions from JSON file if provided
        self.captions = {}
        if self.captions_file:
            with open(self.captions_file, "r") as f:
                captions_data = json.load(f)
                self.captions = {
                    item["image_id"]: item["caption"] for item in captions_data
                }

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        image = Image.open(image_path).convert("RGB")

        # Extract image name without extension
        image_name = os.path.basename(image_path).replace(".png", ".jpg")

        # Retrieve caption or raise an error if not found
        if self.captions_file and image_name in self.captions:
            caption = self.captions[image_name]
            text = caption + " " + self.placeholder_token  # With placeholder token
            text1 = caption  # Without placeholder token
        else:
            raise ValueError(f"Caption not found for image {image_name}")

        # Tokenization
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_ids_1"] = self.tokenizer(
            text1,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # Empty string for null/unconditioned input
        example["input_null"] = self.tokenizer(
            "",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # Convert image to tensor format [C, H, W]
        # example["input_images"] = torch.from_numpy(np.array(image)).permute(2, 0, 1)

        # Apply image transformations if provided
        if self.img_transforms:
            example["pixel_values"] = self.img_transforms(image)

        # Store text prompts for debugging or further processing
        example["text"] = text
        example["text1"] = text1
        example["file_name"] = image_name

        return example

class HuggingFaceTextualInversionDataset(Dataset):
    """
    A dataset class that works with Hugging Face datasets 
    and applies placeholder tokens in text prompts.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        placeholder_token="*",
        img_transforms=None,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.img_transforms = img_transforms
        self._length = len(hf_dataset)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}

        # Get the data item from the Hugging Face dataset
        data_item = self.hf_dataset[i]
        image = Image.open(data_item["image"]).convert("RGB")  # Ensure image is RGB
        caption = data_item["text"]

        # Create the two versions of the text, one with the placeholder token and one without
        text_with_placeholder = caption + " " + self.placeholder_token
        text_without_placeholder = caption

        # Tokenization
        example["input_ids"] = self.tokenizer(
            text_with_placeholder,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_ids_1"] = self.tokenizer(
            text_without_placeholder,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # Empty string for null/unconditioned input
        example["input_null"] = self.tokenizer(
            "",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # Apply image transformations if provided
        if self.img_transforms:
            example["pixel_values"] = self.img_transforms(image)
        else:
            # Default: convert image to tensor format [C, H, W]
            example["pixel_values"] = torch.from_numpy(np.array(image)).permute(2, 0, 1)

        # Store text prompts for debugging or further processing
        example["text"] = text_with_placeholder
        example["text1"] = text_without_placeholder

        return example
    
class TextualInversionDatasetOld(Dataset):
    """
    Works for both watermarking and vanilla Textual Inversion
    """

    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        img_transforms=None,
        captions_file=None,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.img_transforms = img_transforms
        self.captions_file = captions_file

        self.image_paths = [
            os.path.join(self.data_root, file_path)
            for file_path in os.listdir(self.data_root)
        ]
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # Load captions from JSON file if provided
        self.captions = {}
        if self.captions_file:
            with open(self.captions_file, "r") as f:
                captions_data = json.load(f)
                for item in captions_data:
                    self.captions[item["image_id"]] = item["caption"]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # Extract the image name
        image_name = os.path.basename(image_path)
        
        if image_name.endswith(".png"):
            image_name = image_name.replace(".png", ".jpg")

        # Use the caption from the JSON file if available, otherwise use a template
        if self.captions_file is not None:
            if image_name in self.captions:
                text = self.captions[image_name] + " " + self.placeholder_token
                text1 = self.captions[
                    image_name
                ]  # does not contain the placeholder token
            else:
                raise ValueError(f"Caption not found for image {image_name}")
        else:
            placeholder_string = self.placeholder_token
            text = random.choice(self.templates).format(placeholder_string)
            
            # Replace placeholder_string with "cat"
            text1 = text.replace(placeholder_string, "cat toy")
        
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_ids_1"] = self.tokenizer(
            text1,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        null_caption = ""  # This represents the null/unconditioned text
        example["input_null"] = self.tokenizer(
            null_caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        example["input_images"] = torch.from_numpy(np.array(image)).permute(
            2, 0, 1
        )  # [0, 255]
        example["text"] = text
        example["text1"] = text1

        if self.img_transforms is not None:
            image = self.img_transforms(image)
            # print("img_transforms in dataset class range: ", image.min(), image.max())
            # this prints [-1, 1] #checked
            example["pixel_values"] = image

        else:
            img = np.array(image).astype(np.uint8)

            if self.center_crop:
                crop = min(img.shape[0], img.shape[1])
                (
                    h,
                    w,
                ) = (
                    img.shape[0],
                    img.shape[1],
                )
                img = img[
                    (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
                ]

            image = Image.fromarray(img)

            image = image.resize((self.size, self.size), resample=self.interpolation)

            image = self.flip_transform(image)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)

            example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        return example