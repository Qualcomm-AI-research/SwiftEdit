# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPVisionModelWithProjection,
)

from src.mask_ip_controller import *
from src.attention_processor import AttnProcessor2_0 as AttnProcessor
from src.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor
from src.mask_attention_processor import IPAttnProcessor2_0WithIPMaskController

def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(
        self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class InverseModel:
    """
        Inversion Network that bring source image latents to noisy latents.
    """
    def __init__(
        self, 
        pretrained_model_name_path, 
        model_name="stabilityai/sd-turbo",
        dtype="fp32",
        device="cuda"
    ):
        if dtype == "fp16":
            self.weight_dtype = torch.float16
        elif dtype == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32

        self.device = device
        self.model_name = model_name
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae").to(
            self.device, dtype=torch.float32
        )

        self.unet_inverse = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_path, subfolder="unet_ema"
        ).to(self.device, dtype=self.weight_dtype)

        self.unet_inverse.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder"
        ).to(self.device, dtype=self.weight_dtype)

        T = torch.ones((1,), dtype=torch.int64, device=self.device)
        T = T * (self.noise_scheduler.config.num_train_timesteps - 1)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        self.corrupt_alpha_t = (alphas_cumprod[int(T / 4)] ** 0.5).view(-1, 1, 1, 1)
        self.corrupt_sigma_t = ((1 - alphas_cumprod[int(T / 4)]) ** 0.5).view(-1, 1, 1, 1)

        del alphas_cumprod

class AuxiliaryModel:
    """
        A few auxiliary and supported models (text encoder, noise scheduler, tokenizer, ...) as separate modules.
    """
    def __init__(
        self,
        model_name="stabilityai/stable-diffusion-2-1-base",
        image_encoder_path="h94/IP-Adapter",
        device="cuda",
    ):
        self.device = device
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(
            self.device, dtype=torch.float32
        )

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path, subfolder="models/image_encoder"
        ).to(device, dtype=torch.float32)
        self.image_encoder.requires_grad_(False)

        self.clip_image_processor = CLIPImageProcessor()


class IPSBV2Model(torch.nn.Module):
    """
        SwiftBrushv2 model with incorporated IP-Adapter.
    """
    def __init__(
        self,
        pretrained_model_name_path,
        ip_model_path,
        aux_model,
        device="cuda",
        with_ip_mask_controller=False,
    ):
        super().__init__()
        self.device = device
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_path
        ).to(self.device)
        self.unet.eval()
        self.aux_model = aux_model

        self.timestep = torch.ones((1,), dtype=torch.int64, device=self.device)
        self.timestep = self.timestep * (
            self.aux_model.noise_scheduler.config.num_train_timesteps - 1
        )

        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.aux_model.image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        ).to(self.device)

        self.with_ip_mask_controller = with_ip_mask_controller

        # init adapter modules
        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor().to(device)
            else:
                # this is for cross-attention
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                if self.with_ip_mask_controller:
                    attn_procs[name] = IPAttnProcessor2_0WithIPMaskController(
                        hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                    ).to(device)
                else:
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                    ).to(device)
                attn_procs[name].load_state_dict(weights)

        self.unet.set_attn_processor(attn_procs)
        self.adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())

        # prepare stuff
        alphas_cumprod = self.aux_model.noise_scheduler.alphas_cumprod.to(self.device)
        self.alpha_t = (alphas_cumprod[self.timestep] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[self.timestep]) ** 0.5).view(-1, 1, 1, 1)
        del alphas_cumprod

        self.load_state_dict(torch.load(ip_model_path))
        # self.load_ip_adapter(path_ckpt_ip)

    def load_ip_adapter(self, path_ckpt_ip):

        sd = torch.load(path_ckpt_ip, map_location="cpu")
        image_proj_sd = {}
        ip_sd = {}
        for k in sd:
            if k.startswith("unet"):
                pass
            elif k.startswith("image_proj_model"):
                image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
            elif k.startswith("adapter_modules"):
                ip_sd[k.replace("adapter_modules.", "")] = sd[k]

        self.image_proj_model.load_state_dict(image_proj_sd)
        self.adapter_modules.load_state_dict(ip_sd)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.aux_model.clip_image_processor(
                images=pil_image, return_tensors="pt"
            ).pixel_values
            clip_image_embeds = self.aux_model.image_encoder(
                clip_image.to(self.device, dtype=torch.float32)
            ).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float32)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor) or isinstance(
                attn_processor, IPAttnProcessor2_0WithIPMaskController
            ):
                attn_processor.scale = scale

    def set_controller(
        self, controller, where=["down_blocks", "mid_block", "up_blocks"], type_controller=None
    ):

        for name_attn_processor, attn_processor in self.unet.attn_processors.items():
            if isinstance(attn_processor, IPAttnProcessor2_0WithIPMaskController):
                # only set at particular blocks
                for from_where in where:
                    if from_where in name_attn_processor:
                        attn_processor.controller = controller

    @torch.no_grad()
    def gen_img(
        self,
        pil_image=None,
        prompts=None,
        noise=None,
        scale=1.0,
    ):

        self.set_scale(scale)
        num_samples = len(prompts)
        
        # Prepare prompt + image embeds
        if prompts is None:
            prompts = ["best quality, high quality"]

        image_prompt_embeds = self.get_image_embeds(pil_image=pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        input_id = tokenize_captions(self.aux_model.tokenizer, prompts).to(self.device)
        prompt_embeds_ = self.aux_model.text_encoder(input_id)[0]
        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)

        # Feed inverted noise to ip-unet generation
        noise = torch.cat([noise] * num_samples, dim=0)
        model_pred = self.unet(noise, self.timestep, prompt_embeds).sample

        if model_pred.shape[1] == noise.shape[1] * 2:
            model_pred, _ = torch.split(model_pred, noise.shape[1], dim=1)

        pred_original_sample = (noise - self.sigma_t * model_pred) / self.alpha_t

        if self.aux_model.noise_scheduler.config.thresholding:
            pred_original_sample = self.aux_model.noise_scheduler._threshold_sample(
                pred_original_sample
            )
        elif self.aux_model.noise_scheduler.config.clip_sample:
            clip_sample_range = self.aux_model.noise_scheduler.config.clip_sample_range
            pred_original_sample = pred_original_sample.clamp(-clip_sample_range, clip_sample_range)

        pred_original_sample = pred_original_sample / self.aux_model.vae.config.scaling_factor
        image = (
            self.aux_model.vae.decode(pred_original_sample.to(dtype=torch.float32)).sample.float() + 1
        ) / 2

        noise_image = noise / self.aux_model.vae.config.scaling_factor
        noise_image = (
            self.aux_model.vae.decode(noise_image.to(dtype=self.aux_model.vae.dtype)).sample.float()
            + 1
        ) / 2

        return image, noise_image
