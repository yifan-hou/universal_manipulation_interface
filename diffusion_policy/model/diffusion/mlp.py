from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from collections import OrderedDict

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels,
        action_horizon, # used for time encoding
        diffusion_step_embed_dim=256, # used for time ecoding
    ):
        super().__init__()

        # get encoder of time stamp
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed, max_value=action_horizon),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # TO try:
        #   1. residual implementation with feedforward
        #   2. experiment with layers 2~8
        #   3. experiment with hidden layer dimension 512~2048
        self.blocks = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(in_channels, 2048)),
            ('act1', nn.ReLU()),
            ('dense2', nn.Linear(2048, 512)),
            ('act2', nn.ReLU()),
            ('output', nn.Linear(512, out_channels)),
            ('outact', nn.Sigmoid()),
        ]))

        self.diffusion_step_encoder = diffusion_step_encoder
        
    def forward(self,
        x, # (B, :)
        cond, # (B, :)
        timestep, # (1)
    ):
        '''
            returns:
            out : [ batch_size x action_dimension ]
        '''
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(x.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps) # (B, diffusion_step_embed_dim)

        global_feature = torch.cat([
            global_feature, x, cond
        ], axis=-1)

        out = self.blocks(global_feature)
        return out

