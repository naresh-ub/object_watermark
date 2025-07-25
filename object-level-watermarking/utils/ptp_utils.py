# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent

@torch.no_grad()
def text2image_ldm_stable_watermark(
    model,
    prompt: List[str],
    prompt_nw: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input_w = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_nw = model.tokenizer(
        prompt_nw,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings_w = model.text_encoder(text_input_w.input_ids.to(model.device))[0]
    text_embeddings_nw = model.text_encoder(text_input_nw.input_ids.to(model.device))[0]
    
    max_length = text_input_w.input_ids.shape[-1] # both inputs have the same length
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context_w = [uncond_embeddings, text_embeddings_w]
    context_nw = [uncond_embeddings, text_embeddings_nw]
    
    if not low_resource:
        context_w = torch.cat(context_w)
        context_nw = torch.cat(context_nw)
        
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        if t <= 40:
            latents = diffusion_step(model, controller, latents, context_nw, t, guidance_scale, low_resource)
        else:
            latents = diffusion_step(model, controller, latents, context_w, t, guidance_scale, low_resource)
            
    image = latent2image(model.vae, latents)
    return image, latent


# def register_attention_control(model, controller):
#     def ca_forward(self, place_in_unet):
#         to_out = self.to_out
#         if type(to_out) is torch.nn.modules.container.ModuleList:
#             to_out = self.to_out[0]
#         else:
#             to_out = self.to_out
        
#         def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
#             ehc = encoder_hidden_states.clone() if encoder_hidden_states is not None else None
#             hc = hidden_states.clone()

#             is_cross = ehc is not None
            
#             residual = hc

#             if self.spatial_norm is not None:
#                 hc = self.spatial_norm(hc, temb)

#             input_ndim = hc.ndim

#             if input_ndim == 4:
#                 batch_size, channel, height, width = hc.shape
#                 hc = hc.view(batch_size, channel, height * width).transpose(1, 2)

#             batch_size, sequence_length, _ = (
#                 hc.shape if ehc is None else ehc.shape
#             )
#             attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

#             if self.group_norm is not None:
#                 hc = self.group_norm(hc.transpose(1, 2)).transpose(1, 2)

#             query = self.to_q(hc)

#             if ehc is None:
#                 ehc = hc
#             elif self.norm_cross:
#                 ehc = self.norm_ehc(ehc)

#             key = self.to_k(ehc)
#             value = self.to_v(ehc)

#             query = self.head_to_batch_dim(query)
#             key = self.head_to_batch_dim(key)
#             value = self.head_to_batch_dim(value)

#             attention_probs = self.get_attention_scores(query, key, attention_mask)
#             attention_probs = controller(attention_probs, is_cross, place_in_unet)

#             hc = torch.bmm(attention_probs, value)
#             hc = self.batch_to_head_dim(hc)

#             # linear proj
#             hc = to_out(hc)

#             if input_ndim == 4:
#                 hc = hc.transpose(-1, -2).reshape(batch_size, channel, height, width)

#             if self.residual_connection:
#                 hc = hc + residual

#             hc = hc / self.rescale_output_factor

#             return hc
#         return forward

#     class DummyController:

#         def __call__(self, *args):
#             return args[0]

#         def __init__(self):
#             self.num_att_layers = 0

#     if controller is None:
#         controller = DummyController()

#     def register_recr(net_, count, place_in_unet):
#         if net_.__class__.__name__ == 'Attention':
#             net_.forward = ca_forward(net_, place_in_unet)
#             return count + 1
#         elif hasattr(net_, 'children'):
#             for net__ in net_.children():
#                 count = register_recr(net__, count, place_in_unet)
#         return count

#     cross_att_count = 0
#     sub_nets = model.unet.named_children()
#     for net in sub_nets:
#         if "down" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "down")
#         elif "up" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "up")
#         elif "mid" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "mid")

#     controller.num_att_layers = cross_att_count

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if isinstance(to_out, torch.nn.modules.container.ModuleList):
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def reshape_heads_to_batch_dim(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor1 = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            tensor2 = tensor1.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
            return tensor2

        def reshape_batch_dim_to_heads(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor1 = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
            tensor2 = tensor1.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
            return tensor2

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            # x = hidden_states.clone()
            # # print("x requires grad?", x.requires_grad) True
            # context = encoder_hidden_states
            # mask = attention_mask
            
            # batch_size, sequence_length, dim = x.shape
            # h = self.heads
            # q = self.to_q(x)
            # is_cross = context is not None
            # context1 = context.clone() if is_cross else x.clone()

            # k = self.to_k(context1)
            # v = self.to_v(context1)

            # q1 = reshape_heads_to_batch_dim(self, q)
            # k1 = reshape_heads_to_batch_dim(self, k)
            # v1 = reshape_heads_to_batch_dim(self, v)

            # sim = torch.einsum("b i d, b j d -> b i j", q1, k1) * self.scale

            # if mask is not None:
            #     mask = mask.reshape(batch_size, -1)
            #     max_neg_value = -torch.finfo(sim.dtype).max
            #     mask = mask[:, None, :].repeat(h, 1, 1)
            #     sim = sim.masked_fill(~mask, max_neg_value)

            # attn = sim.softmax(dim=-1)
            # # Ensure 'attn' is not modified in place by controller or subsequent code
            # attn1 = controller(attn.clone(), is_cross, place_in_unet)  # Cloning here to avoid in-place modification issues

            # out = torch.einsum("b i j, b j d -> b i d", attn1, v1)
            # # print("out requires grad?", out.requires_grad)  # True
            # out1 = reshape_batch_dim_to_heads(self, out.clone())
            # return to_out(out1)
            
            ####### to solve in-place modification issue
            
            x = hidden_states.clone()
            # print("x requires grad?", x.requires_grad) True
            context = encoder_hidden_states
            mask = attention_mask
            
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context1 = context.clone() if is_cross else x.clone()

            k = self.to_k(context1)
            v = self.to_v(context1)

            q1 = reshape_heads_to_batch_dim(self, q)
            k1 = reshape_heads_to_batch_dim(self, k)
            v1 = reshape_heads_to_batch_dim(self, v)

            sim = torch.einsum("b i d, b j d -> b i j", q1, k1) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim = sim.masked_fill(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # Ensure 'attn' is not modified in place by controller or subsequent code
            attn1 = controller(attn.clone(), is_cross, place_in_unet)  # Cloning here to avoid in-place modification issues

            # Cloning to ensure `attn1` is not modified in-place
            out = torch.einsum("b i j, b j d -> b i d", attn1.clone(), v1.clone())
            
            # print("out requires grad?", out.requires_grad)  # True
            out1 = reshape_batch_dim_to_heads(self, out.clone())
            return to_out(out1)
        return forward


    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet, module_name=None):
        # if net_.__class__.__name__ == 'CrossAttention':
        #     net_.forward = ca_forward(net_, place_in_unet)
        #     return count + 1
        if module_name in ["attn1", "attn2"]:
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for k,net__ in net_.named_children():
                count = register_recr(net__, count, place_in_unet, module_name = k)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words

def view_images(images, num_rows=1, offset_ratio=0.02, save_path=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    
    if save_path:
        pil_img.save(save_path)
    else:
        display(pil_img)
        
def view_images1(images, num_rows=1, offset_ratio=0.02, save_path=None):
    
    print("images shape is ", images.shape)
    # Ensure images is a list
    if type(images) is not list:
        images = [images]
        
    print(len(images))
        
    # Calculate the difference image between the first and second images
    if len(images) >= 2:
        difference_image = np.abs(images[0].astype(np.int16) - images[1].astype(np.int16))
        difference_image = np.clip(difference_image, 0, 255).astype(np.uint8)
        images.append(difference_image)
    else:
        raise ValueError("At least two images are required to compute the difference.")

    # Calculate the number of empty images needed for alignment
    num_empty = len(images) % num_rows

    # Prepare empty images for alignment
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)
    
    print("Number of images is", num_items)

    # Determine image properties
    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows

    # Create a blank canvas for displaying the images
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255

    # Paste the images onto the canvas
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    # Convert to PIL Image and display or save
    pil_img = Image.fromarray(image_)
    
    if save_path:
        pil_img.save(save_path)
    else:
        display(pil_img)

