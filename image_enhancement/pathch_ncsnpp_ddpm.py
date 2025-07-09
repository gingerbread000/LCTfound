# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

from . import utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np
import pdb
from diffusers import UNet2DModel, UNet2DConditionModel

import safetensors

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ncsnpp_ddpm')
class NCSNpp_DDPM(nn.Module):
  """NCSN+ model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = config.model.conditional  # noise-conditional
    
    self.skip_rescale = skip_rescale = config.model.skip_rescale
    self.resblock_type = resblock_type = config.model.resblock_type.lower()
    self.progressive = progressive = config.model.progressive.lower()
    self.progressive_input = progressive_input = config.model.progressive_input.lower()
    self.embedding_type = embedding_type = config.model.embedding_type.lower()
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
  
    self.unet = UNet2DConditionModel(
            sample_size=256,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(64, 128, 256, 512, 1024),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", 
                "CrossAttnUpBlock2D",  
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            norm_num_groups=32,
            # addition_embed_type="text",
            # addition_embed_type_num_heads=64,
            encoder_hid_dim_type="text_proj",
            encoder_hid_dim=768,
        )

    model_path = "/path/to/pretrained/model/"
    pretrained_w = safetensors.torch.load_file(model_path + "/diffusion_pytorch_model.safetensors")
    
    self.unet.load_state_dict(pretrained_w, strict=False)
    print(f"load from {model_path}")
    self.unet.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    self.unet.conv_in = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    torch.nn.init.kaiming_normal_(self.unet.conv_out.weight)
    torch.nn.init.kaiming_normal_(self.unet.conv_in.weight)

  def forward(self, x, time_cond, prompt_emb=None):
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond

    if not self.config.data.centered:
      x = 2 * x - 1.

    h = self.unet(x, time_cond, prompt_emb=prompt_emb)[0]
    # pdb.set_trace()

    if self.config.model.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      # debug
      # print(f'used_sigmas: {used_sigmas.shape}')
      h = h / used_sigmas
    return h
