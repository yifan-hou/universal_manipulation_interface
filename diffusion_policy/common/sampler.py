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
        replay_buffer: dict,
        key_horizon: dict, # obs: observation horizon, action: action horizon
        key_down_sample_steps: dict,
        query_frequency_down_sample_steps: int=1,
        episode_mask: Optional[np.ndarray]=None,
        action_padding: bool=False,
        obs_to_obs_sample=None,
        action_to_action_sample=None,
    ):
        # Computes indices. indices[i] = (epi_id, epi_len, id)
        #   epi_id: which episode the index i belongs to.
        #   epi_len: length of the episode.
        #   id: the index within the episode.
        # here, index i is assumed to be using the low dim index.
        episodes_length = replay_buffer['meta']['episode_low_dim_len'][:]
        episodes_length_for_query = episodes_length.copy()
        if not action_padding:
            # if no action padding, truncate the indices to query so the last query point
            #  still has access to the whole horizon of actions
            episodes_length_for_query -= (key_horizon['action'] - 1) * key_down_sample_steps['action']
        assert(np.min(episodes_length) > 0)
        epi_id = []
        epi_len = []
        ids = []
        for array_index, array_length in enumerate(episodes_length_for_query):
            if episode_mask is not None and not episode_mask[array_index]:
                # skip episode
                continue
            epi_id.extend([array_index] * array_length)
            epi_len.extend([episodes_length[array_index]] * array_length)
            ids.extend(range(array_length))
            # assert(epi_len[-1] >= ids[-1] + (key_horizon['action'] - 1) * key_down_sample_steps['action'] + 1)

        N_indices = len(epi_id)
        epi_id = epi_id[::query_frequency_down_sample_steps]
        epi_len = epi_len[::query_frequency_down_sample_steps]
        ids = ids[::query_frequency_down_sample_steps]

        indices = list(zip(epi_id, epi_len, ids))

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.action_padding = action_padding
        self.indices = indices
        self.key_horizon = key_horizon
        self.key_down_sample_steps = key_down_sample_steps
        self.obs_to_obs_sample = obs_to_obs_sample
        self.action_to_action_sample = action_to_action_sample

        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizer

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        epi_id, epi_len, id = self.indices[idx]
        episode = f'episode_{epi_id}'
        data_episode = self.replay_buffer['data'][episode]
        obs_dict = dict()
        action_array = []

        # observation
        #   step one: read correct length
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            input_arr = data_episode[key]
            this_horizon = self.key_horizon[key]
            this_downsample_steps = self.key_down_sample_steps[key]

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                if self.ignore_rgb_is_applied:
                    continue
                num_valid = min(this_horizon, id // this_downsample_steps + 1)
                slice_start = id - (num_valid - 1) * this_downsample_steps
                assert(slice_start >= 0)
                output = input_arr[slice_start: id + 1: this_downsample_steps]
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
                    [id - i * this_downsample_steps for i in range(this_horizon)])
                idx_in_obs_horizon = idx_in_obs_horizon[::-1] # reverse order, so ids are increasing
                idx_in_obs_horizon = np.clip(idx_in_obs_horizon, 0, id)

                output = input_arr[idx_in_obs_horizon].astype(np.float32)
            obs_dict[key] = output

        #   step two: convert to relative pose
        obs_sample = self.obs_to_obs_sample(obs_dict)

        # action
        #   step one: read correct length
        input_arr = data_episode['action']
        action_horizon = self.key_horizon['action']
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(epi_len, id + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[id: slice_end: action_down_sample_steps]
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        #   step two: convert to relative pose
        action_sample = self.action_to_action_sample(output)

        return obs_sample, action_sample

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply