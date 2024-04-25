import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
# import shutil

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
# from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from umi.common.pose_util import pose7_to_mat, mat_to_pose9d

register_codecs()

class FlipUpDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        seed: int=42,
        val_ratio: float=0.0,
    ):
        # load into memory store
        print('[FlipUpDataset] loading data into store')
        with zarr.DirectoryStore(dataset_path) as directory_store:
            replay_buffer_raw = ReplayBuffer.copy_from_store(
                src_store=directory_store,
                dest_store=zarr.MemoryStore()
            )
        print('[FlipUpDataset] raw to obs/action conversion')
        replay_buffer = self.raw_to_obs_action(replay_buffer_raw, shape_meta)

        print('[FlipUpDataset] reading meta info')
        obs_rgb_keys = list()
        obs_lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # obtain obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_rgb_keys.append(key)
            elif type == 'low_dim':
                obs_lowdim_keys.append(key)

            # obtain obs_horizon info
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # obtain down_sample_steps info
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # obtain action info
        key_horizon['action'] = shape_meta['action']['horizon']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        # train/val mask for training
        val_mask = get_val_mask(
            n_episodes=replay_buffer_raw.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        print('[FlipUpDataset] creating SequenceSampler.')
        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            key_horizon=key_horizon,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
        )

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.obs_rgb_keys = obs_rgb_keys
        self.obs_lowdim_keys = obs_lowdim_keys
        self.key_horizon = key_horizon
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

    def raw_to_obs_action(self,
        replay_buffer_raw: ReplayBuffer,
        shape_meta: dict
    ):
        replay_buffer = dict()
        replay_buffer['data'] = dict()

        for ep in replay_buffer_raw['data'].keys():
            # iterates over episodes
            # ep: 'episode_xx'
            replay_buffer['data'][ep] = dict()
            # obs.rgb: keep entry, keep as compressed zarr array in memory
            for key, attr in shape_meta['raw'].items():
                type = attr.get('type', 'low_dim')
                if type == 'rgb':
                    # obs.rgb: keep as compressed zarr array in memory
                    replay_buffer['data'][ep][key] = replay_buffer_raw['data'][ep][key]

            # obs.low_dim: load entry, convert to obs.low_dim
            ts_pose_fb = replay_buffer_raw['data'][ep]['ts_pose_fb']
            ts_pose_fb_9d = mat_to_pose9d(pose7_to_mat(ts_pose_fb))

            replay_buffer['data'][ep]['robot0_eef_pos'] = ts_pose_fb_9d[:, :3]
            replay_buffer['data'][ep]['robot0_eef_rot_axis_angle'] = ts_pose_fb_9d[:, 3:]
            replay_buffer['data'][ep]['robot0_eef_wrench'] = replay_buffer_raw['data'][ep]['wrench'][:]

            # action: assemble from low_dim
            ts_pose_command = replay_buffer_raw['data'][ep]['ts_pose_command']
            ts_pose_command_9d = mat_to_pose9d(pose7_to_mat(ts_pose_fb))
            replay_buffer['data'][ep]['action'] = ts_pose_command_9d[:]

        # meta
        replay_buffer['meta'] = replay_buffer_raw['meta']
        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            key_horizon=self.key_horizon,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    ##
    ## Read all action data, compute normalizer parameters from them, return the
    ## initialized normalizers.
    ##
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.obs_lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.obs_lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        # One key [pos, rot]
        # needs to change for multiple robots. See umi_dataset.py
        action_normalizers = list()
        action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., 0:3])))   # pos
        action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., 3:]))) # rot
        # action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., (i + 1) * dim_a - 1: (i + 1) * dim_a])))  # gripper

        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.obs_lowdim_keys:
            stat = array_to_stats(data_cache[key])
            if 'eef_pos' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif 'rot_axis_angle' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif 'wrench' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.obs_rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        obs_dict, action_array = self.sampler.sample_sequence(idx)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action_array.astype(np.float32))
        }
        return torch_data
