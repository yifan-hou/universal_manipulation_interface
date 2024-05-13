from typing import Optional, Callable
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
    """ Sample sequences of observations and actions from replay buffer.
        Query ID is based on rgb data, which is likely to be the most sparse data.
        Other data corresponding to the query ID is obtain based on timestamps.
        1. Given query id, find the corresponding low dim/rgb id.
        1. Construct sparse sample:
            Sample sparse obs horizon before idx,
            Sample sparse action horizon after idx.
        2. Construct dense sample:
            Find the indices of dense query points. For each dense query point:
                Sample dense obs horizon before idx,
                Sample dense action horizon after idx.
    """
    def __init__(self,
        shape_meta:dict,
        replay_buffer: dict,
        sparse_query_frequency_down_sample_steps: int=1,
        dense_query_frequency_down_sample_steps: int=1,
        episode_mask: Optional[np.ndarray]=None,
        action_padding: bool=False,
        obs_to_obs_sample: Optional[Callable]=None,
        action_to_action_sample: Optional[Callable]=None,
    ):
        episode_keys = replay_buffer['data'].keys()
        # Step one: Find the usable length of each episode
        episodes_length = replay_buffer['meta']['episode_rgb_len'][:]
        episodes_length_for_query = episodes_length.copy()
        if not action_padding:
            # if no action padding, truncate the indices to query so the last query point
            #  still has access to the whole horizon of actions
            #  This is enforced by sparse action alone. It is assumed that the dense action is
            #  not affected.
            sparse_action_horizon = shape_meta['sample']['action']['sparse']['horizon']
            sparse_action_down_sample_steps = shape_meta['sample']['action']['sparse']['down_sample_steps']
            action_chopped_len = (sparse_action_horizon - 1) * sparse_action_down_sample_steps
            episode_count = -1
            for episode in episode_keys:
                episode_count += 1
                low_dim_end_time = replay_buffer['data'][episode]['action_time_stamps'][-action_chopped_len-1]
                rgb_times = replay_buffer['data'][episode]['obs']['visual_time_stamps']
                # find the last rgb_times index that is before low_dim_end_time
                last_rgb_idx = np.searchsorted(rgb_times, low_dim_end_time, side='right') - 1
                episodes_length_for_query[episode_count] = last_rgb_idx
        assert(np.min(episodes_length) > 0)

        # Step two: Computes indices from episodes_length_for_query. indices[i] = (epi_id, epi_len, id)
        #   epi_id: which episode the index i belongs to.
        #   epi_len: length of the episode.
        #   id: the index within the episode.
        epi_id = []
        epi_len = []
        ids = []
        episode_count = -1
        for key in episode_keys:
            episode_count += 1
            episode_index = int(key.split('_')[-1])
            array_length = episodes_length_for_query[episode_count]
            if episode_mask is not None and not episode_mask[episode_count]:
                # skip episode
                continue
            epi_id.extend([episode_index] * array_length)
            epi_len.extend([episodes_length[episode_count]] * array_length)
            ids.extend(range(array_length))

        # Step three: Down sample the query indices to make the dataset smaller
        epi_id = epi_id[::sparse_query_frequency_down_sample_steps]
        epi_len = epi_len[::sparse_query_frequency_down_sample_steps]
        ids = ids[::sparse_query_frequency_down_sample_steps]

        indices = list(zip(epi_id, epi_len, ids))

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.action_padding = action_padding
        self.indices = indices
        self.obs_to_obs_sample = obs_to_obs_sample
        self.action_to_action_sample = action_to_action_sample
        self.dense_query_frequency_down_sample_steps = dense_query_frequency_down_sample_steps

        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizer

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        """ Sample a sequence of observations and actions at idx.

        """
        epi_id, epi_len_rgb, rgb_id = self.indices[idx]
        episode = f'episode_{epi_id}'
        data_episode = self.replay_buffer['data'][episode]

        has_dense = True if 'dense' in self.shape_meta['sample']['obs'].keys() else False

        # indices are for the rgb obs data. To get low dim obs and action, we need to find their id
        rgb_time = data_episode['obs']['visual_time_stamps'][rgb_id]
        low_dim_id = np.searchsorted(data_episode['obs']['low_dim_time_stamps'], rgb_time)
        action_id = np.searchsorted(data_episode['action_time_stamps'], rgb_time)

        # sparse obs
        sparse_obs_unprocessed = dict()
        for key, attr in self.shape_meta['sample']['obs']['sparse'].items():
            input_arr = data_episode['obs'][key]
            this_horizon = attr['horizon']
            this_downsample_steps = attr['down_sample_steps']
            type = self.shape_meta['obs'][key]['type']

            if type == 'rgb':
                id = rgb_id
            elif type == 'low_dim':
                id = low_dim_id

            num_valid = min(this_horizon, id // this_downsample_steps + 1)
            slice_start = id - (num_valid - 1) * this_downsample_steps
            assert(slice_start >= 0)
            
            if type == 'rgb':
                if self.ignore_rgb_is_applied:
                    continue
                output = input_arr[slice_start: id + 1: this_downsample_steps]
            elif type == 'low_dim':
                output = input_arr[slice_start: id + 1: this_downsample_steps].astype(np.float32)
            assert output.shape[0] == num_valid
            # solve padding
            if output.shape[0] < this_horizon:
                padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                output = np.concatenate([padding, output], axis=0)
            sparse_obs_unprocessed[key] = output
        
        # sparse action
        input_arr = data_episode['action']
        action_horizon = self.shape_meta['sample']['action']['sparse']['horizon']
        action_down_sample_steps = self.shape_meta['sample']['action']['sparse']['down_sample_steps']
        slice_end = min(len(input_arr)-1, action_id + (action_horizon - 1) * action_down_sample_steps + 1)
        sparse_action_unprocessed = input_arr[action_id: slice_end: action_down_sample_steps]
        # solve padding
        if not self.action_padding:
            assert sparse_action_unprocessed.shape[0] == action_horizon
        elif sparse_action_unprocessed.shape[0] < action_horizon:
            padding = np.repeat(sparse_action_unprocessed[-1:], action_horizon - sparse_action_unprocessed.shape[0], axis=0)
            sparse_action_unprocessed = np.concatenate([sparse_action_unprocessed, padding], axis=0)

        dense_obs_unprocessed = {}
        dense_action_unprocessed = []
        if has_dense:
            sparse_action_horizon = self.shape_meta['sample']['action']['sparse']['horizon']
            sparse_action_down_sample_steps = self.shape_meta['sample']['action']['sparse']['down_sample_steps']

            a_dense_obs_key = next(iter(self.shape_meta['sample']['obs']['dense'].values()))
            dense_obs_horizon = a_dense_obs_key['horizon']
            dense_obs_down_sample_steps = a_dense_obs_key['down_sample_steps']
            dense_action_horizon = self.shape_meta['sample']['action']['dense']['horizon']
            dense_action_down_sample_steps = self.shape_meta['sample']['action']['dense']['down_sample_steps']
            
            dense_action_num_of_queries = sparse_action_horizon * sparse_action_down_sample_steps \
                - dense_action_horizon * dense_action_down_sample_steps
            dense_queries = np.arange(0, dense_action_num_of_queries, self.dense_query_frequency_down_sample_steps)

            dense_queries_obs_start = dense_queries - (dense_obs_horizon-1) * dense_obs_down_sample_steps
            dense_queries_action_end = dense_queries + (dense_action_horizon-1) * dense_action_down_sample_steps

            # dense obs (H, T, D)
            # assuming no padding is needed
            for key in self.shape_meta['sample']['obs']['dense'].keys():
                input_arr = data_episode['obs'][key]
                output = input_arr[[np.arange(dense_queries_obs_start[i], dense_queries[i]+1, dense_obs_down_sample_steps) for i in range(len(dense_queries))]]
                dense_obs_unprocessed[key] = output.astype(np.float32)

            # dense action (H, T, D)
            # assuming no padding is needed
            dense_action_unprocessed = data_episode['action'][[np.arange(dense_queries[i], dense_queries_action_end[i]+1, dense_action_down_sample_steps) for i in range(len(dense_queries))]]

        #   convert to relative pose
        obs_sample = self.obs_to_obs_sample(sparse_obs_unprocessed, dense_obs_unprocessed, self.shape_meta, 'check', self.ignore_rgb_is_applied)
        action_sample = self.action_to_action_sample(sparse_action_unprocessed, dense_action_unprocessed)

        return obs_sample, action_sample


    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply
