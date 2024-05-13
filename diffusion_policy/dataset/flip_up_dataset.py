import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, '../../../'))

import copy
from typing import Dict, Optional

import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
from einops import rearrange, reduce

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

import sys
import os

from PyriteConfig.tasks.flip_up.flip_up_type_conversion import (
    raw_to_obs, raw_to_action, obs_to_obs_sample, action_to_action_sample)

register_codecs()

class FlipUpDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        sparse_query_frequency_down_sample_steps: int=1,
        dense_query_frequency_down_sample_steps: int=1,
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
        # convert raw to replay buffer
        print('[FlipUpDataset] raw to obs/action conversion')
        replay_buffer = self.raw_episodes_conversion(replay_buffer_raw, shape_meta)

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
            sparse_query_frequency_down_sample_steps=sparse_query_frequency_down_sample_steps,
            dense_query_frequency_down_sample_steps=dense_query_frequency_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            obs_to_obs_sample=obs_to_obs_sample,
            action_to_action_sample=action_to_action_sample,
        )

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.sparse_query_frequency_down_sample_steps = sparse_query_frequency_down_sample_steps
        self.dense_query_frequency_down_sample_steps = dense_query_frequency_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

    def raw_episodes_conversion(self,
        replay_buffer_raw: ReplayBuffer,
        shape_meta: dict
    ):
        replay_buffer = dict()
        replay_buffer['data'] = dict()

        for ep in replay_buffer_raw['data'].keys():
            # iterates over episodes
            # ep: 'episode_xx'
            replay_buffer['data'][ep] = dict()
            raw_to_obs(replay_buffer_raw['data'][ep], replay_buffer['data'][ep], shape_meta)
            raw_to_action(replay_buffer_raw['data'][ep], replay_buffer['data'][ep])

        # meta
        replay_buffer['meta'] = replay_buffer_raw['meta']
        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            sparse_query_frequency_down_sample_steps=self.sparse_query_frequency_down_sample_steps,
            dense_query_frequency_down_sample_steps=self.dense_query_frequency_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            obs_to_obs_sample=obs_to_obs_sample,
            action_to_action_sample=action_to_action_sample,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> tuple:
        """ Compute normalizer for each key in the dataset.
            Note: only low_dim and action are considered. Image does not need normalization.
            return: tuple of normalizers for sparse and dense obs. Dense one might be None.
        """
        sparse_normalizer = LinearNormalizer()
        dense_normalizer = LinearNormalizer()

        no_dense_key = False
        if len(self.shape_meta['sample']['obs'].keys()) == 1:
            # there is no dense key
            no_dense_key = True

        # gather all data in cache
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        data_cache_sparse = {}
        data_cache_dense = {}
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            # sparse obs
            for key in self.shape_meta['sample']['obs']['sparse'].keys():
                if self.shape_meta['obs'][key]['type'] == 'low_dim':
                    if key not in data_cache_sparse.keys():
                        data_cache_sparse[key] = []
                    data_cache_sparse[key].append(copy.deepcopy(batch['obs']['sparse'][key]))
            if 'action' not in data_cache_sparse.keys():
                data_cache_sparse['action'] = []
            data_cache_sparse['action'].append(copy.deepcopy(batch['action']['sparse']))
            # dense obs
            if no_dense_key:
                continue
            for key in self.shape_meta['sample']['obs']['dense'].keys():
                if self.shape_meta['obs'][key]['type'] == 'low_dim':
                    if key not in data_cache_dense.keys():
                        data_cache_dense[key] = []
                    data_cache_dense[key].append(copy.deepcopy(batch['obs']['dense'][key]))
            if 'action' not in data_cache_dense.keys():
                data_cache_dense['action'] = []
            data_cache_dense['action'].append(copy.deepcopy(batch['action']['dense']))
        self.sampler.ignore_rgb(False)

        for data_cache in [data_cache_sparse, data_cache_dense]:
            for key in data_cache.keys():
                data_cache[key] = np.concatenate(data_cache[key])
                if not self.temporally_independent_normalization:
                    data_cache[key] = rearrange(data_cache[key], 'B T ... -> (B T) (...)')

        # sparse: compute normalizer for action
        sparse_action_normalizers = list()
        sparse_action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache_sparse['action'][..., 0:3])))   # pos
        sparse_action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache_sparse['action'][..., 3:]))) # rot
        sparse_normalizer['action'] = concatenate_normalizer(sparse_action_normalizers)

        # sparse: compute normalizer for obs
        for key in self.shape_meta['sample']['obs']['sparse'].keys():
            type = self.shape_meta['obs'][key]['type']
            if type == 'low_dim':
                stat = array_to_stats(data_cache_sparse[key])
                if 'eef_pos' in key:
                    this_normalizer = get_range_normalizer_from_stat(stat)
                elif 'rot_axis_angle' in key:
                    this_normalizer = get_identity_normalizer_from_stat(stat)
                elif 'wrench' in key:
                    this_normalizer = get_range_normalizer_from_stat(stat)
                else:
                    raise RuntimeError('unsupported')
                sparse_normalizer[key] = this_normalizer
            elif type == 'rgb':
                sparse_normalizer[key] = get_image_identity_normalizer()
            else:
                raise RuntimeError('unsupported')

        if no_dense_key:
            return sparse_normalizer, None

        # dense: compute normalizer for action
        dense_action_normalizers = list()
        dense_action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache_dense['action'][..., 0:3])))   # pos
        dense_action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache_dense['action'][..., 3:]))) # rot
        dense_normalizer['action'] = concatenate_normalizer(dense_action_normalizers)

        # dense: compute normalizer for obs
        for key in self.shape_meta['sample']['obs']['dense'].keys():
            stat = array_to_stats(data_cache_dense[key])
            if 'eef_pos' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif 'rot_axis_angle' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif 'wrench' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            dense_normalizer[key] = this_normalizer

        return sparse_normalizer, dense_normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        obs_dict, action_array = self.sampler.sample_sequence(idx)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': dict_apply(action_array, torch.from_numpy)
        }
        return torch_data
