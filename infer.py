# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os, time

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from models import *

#
# Configure this path to where you have stored the local copy of the weights:
#
SWIFTEDIT_WEIGHTS_ROOT = 'swiftedit_weights'

def to_binary(pix, threshold=0.5):
    if float(pix) > threshold:
        return 1.0
    else:
        return 0.0


@torch.no_grad()
def edit_image(
    img_path,
    src_p,
    edit_p,
    inverse_model,
    aux_model,
    ip_sb_model,
    scale_ta=1,
    scale_edit=0.2,
    scale_non_edit=1,
    clamp_rate=3.0,
    mask_threshold=0.5,
):
    """
        Save keysteps to file.
            + img_path: path to the source image.
            + src_p: Source Prompt that describes source image (could leave it empty).
            + edit_p: Edit Prompt that describes your desired changes.
    """
    mid_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 500
    final_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 999

    # Input Image
    pil_img_cond = Image.open(img_path).convert("RGB").resize((512, 512))

    processed_image = to_tensor(pil_img_cond).unsqueeze(0).to("cuda") * 2 - 1

    # Predict inverted noise
    latents = inverse_model.vae.encode(
        processed_image.to(inverse_model.weight_dtype)
    ).latent_dist.sample()
    latents = latents * inverse_model.vae.config.scaling_factor
    dub_latents = torch.cat([latents] * 2, dim=0)

    input_id = tokenize_captions(inverse_model.tokenizer, [src_p, edit_p]).to("cuda")
    encoder_hidden_state = inverse_model.text_encoder(input_id)[0].to(
        dtype=inverse_model.weight_dtype
    )

    predict_inverted_code = inverse_model.unet_inverse(
        dub_latents, mid_timestep, encoder_hidden_state
    ).sample.to("cuda", dtype=inverse_model.weight_dtype)

    # Estimate editing mask
    inverted_noise_1, inverted_noise_2 = predict_inverted_code.chunk(2)
    subed = (inverted_noise_1 - inverted_noise_2).abs_().mean(dim=[0, 1])
    max_v = (subed.mean() * clamp_rate).item()
    mask12 = subed.clamp(0, max_v) / max_v
    mask12 = mask12.detach().cpu().apply_(lambda pix: to_binary(pix, mask_threshold)).to("cuda")

    # Edit images
    input_sb = ip_sb_model.alpha_t * latents + ip_sb_model.sigma_t * inverted_noise_1
    mask_controller = MaskController(
        mask12, scale_text_hiddenstate=scale_ta, scale_ip_fg=scale_edit, scale_ip_bg=scale_non_edit
    )
    ip_sb_model.set_controller(mask_controller, where=["mid_blocks", "up_blocks"])
    res_gen_img, _ = ip_sb_model.gen_img(
        pil_image=pil_img_cond, prompts=[src_p, edit_p], noise=input_sb
    )

    return res_gen_img


if __name__ == "__main__":

    # Define model
    inverse_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "inverse_ckpt-120k")
    inverse_model = InverseModel(inverse_ckpt)
    aux_model = AuxiliaryModel()

    path_unet_sb = (os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "sbv2_0.5"))
    ip_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "ip_adapter_ckpt-90k/ip_adapter.bin")
    ip_sb_model = IPSBV2Model(path_unet_sb, ip_ckpt, aux_model, with_ip_mask_controller=True)

    # Input

    img_path = "./assets/imgs_demo/woman_face.jpg"
    src_p = "woman"
    edit_p = "Taylor Swift"

    # img_path = "./assets/imgs_demo/02.jpg"
    # src_p = "dog"
    # edit_p = "dog with mouth opened"

    start_time = time.time()
    result = edit_image(img_path, src_p, edit_p, inverse_model, aux_model, ip_sb_model)
    print(f"Edit {src_p}->{edit_p} in {time.time()-start_time}")
    save_image(result, f"result_{src_p}->{edit_p}.png")