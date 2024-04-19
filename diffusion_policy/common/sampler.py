from typing import Optional
import numpy as np
import random
import scipy.interpolate as si
import scipy.spatial.transform as st
from diffusion_policy.common.replay_buffer import ReplayBuffer

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class SequenceSampler:
    def __init__(self,
        shape_meta:dict,
        replay_buffer_raw: ReplayBuffer,
        replay_buffer: dict,
        key_horizon: dict, # obs: observation horizon, action: action horizon
        key_down_sample_steps: dict,
        episode_mask: Optional[np.ndarray]=None,
        action_padding: bool=False,
    ):
        episode_ends = replay_buffer_raw.episode_ends[:]

        # create indices, including (current_idx, start_idx, end_idx)
        # Note that all episodes are concantenated. episode_ends is a list of end indices for all episodes.
        # indices describes which episode a certain id belongs to.
        # for i, indices[i] has the following info:
        #   current_idx: query id. Equals to i
        #   start_idx: idx of the beginning of the episode that contains i
        #   end_idx: idx of the end of the episode that contains i
        indices = list()
        for i in range(len(episode_ends)):
            if episode_mask is not None and not episode_mask[i]:
                # skip episode
                continue
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            for current_idx in range(start_idx, end_idx):
                if not action_padding and end_idx < current_idx + (key_horizon['action'] - 1) * key_down_sample_steps['action'] + 1:
                    continue
                indices.append((current_idx, start_idx, end_idx))

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.action_padding = action_padding
        self.indices = indices
        self.key_horizon = key_horizon
        self.key_down_sample_steps = key_down_sample_steps

        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizaer

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        current_idx, start_idx, end_idx = self.indices[idx]
        obs_dict = dict()
        action_array = []

        # observation
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            input_arr = self.replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_downsample_steps = self.key_down_sample_steps[key]

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                if self.ignore_rgb_is_applied:
                    continue
                num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)
                slice_start = current_idx - (num_valid - 1) * this_downsample_steps

                output = input_arr[slice_start: current_idx + 1: this_downsample_steps]
                assert output.shape[0] == num_valid

                # solve padding
                if output.shape[0] < this_horizon:
                    padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)
                # move channel last to channel first
                # T,H,W,C
                # convert uint8 image to float32
                output = np.moveaxis(output, -1, 1).astype(np.float32) / 255.
            elif type == 'low_dim':
                idx_in_obs_horizon = np.array(
                    [current_idx - idx * this_downsample_steps for idx in range(this_horizon)],
                    dtype=np.float32)
                idx_in_obs_horizon = idx_in_obs_horizon[::-1] # reverse order, so ids are increasing
                idx_in_obs_horizon = np.clip(idx_in_obs_horizon, start_idx, end_idx - 1)

                output = input_arr[idx_in_obs_horizon].astype(np.float32)

        obs_dict[key] = output

        # action
        input_arr = self.replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_idx: slice_end: action_down_sample_steps]
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        action_array = output

        return obs_dict, action_array

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply