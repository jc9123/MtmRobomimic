# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset used for training a policy. Formed from a collection of
HDF5 files and wrapped into a PyTorch Dataset.
"""

from typing import Any, Callable, Dict, Sequence, Tuple
import json

import numpy as np
import torch
from research.jaxrl.datasets.dataset import Dataset
from dataclasses import dataclass
import gym
from collections import defaultdict

import h5py  # Import the h5py library to handle HDF5 files
import tqdm
import wandb

from research.mtm.datasets.base import DatasetProtocol, DataStatistics
from research.mtm.tokenizers.base import TokenizerManager
from research.robomimic.dummy import test_sanity
from research.robomimic.dummy import printer
import research.robomimic.robomimic.utils.env_utils as FileUtils
import research.robomimic.robomimic.utils.env_utils as EnvUtils
import research.robomimic.robomimic.utils.obs_utils as ObsUtils
from research.robomimic.robomimic.config import config_factory

import os

@dataclass(frozen=True)
class Trajectory:
    """Immutable container for a Trajectory.

    Each has shape (T, X).
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    def __post_init__(self):
        assert self.observations.shape[0] == self.actions.shape[0]
        assert self.observations.shape[0] == self.rewards.shape[0]

    def __len__(self) -> int:
        return self.observations.shape[0]

    @staticmethod
    def create_empty(
        observation_shape: Sequence[int], action_shape: Sequence[int]
    ) -> "Trajectory":
        """Create an empty trajectory."""
        return Trajectory(
            observations=np.zeros((0,) + observation_shape),
            actions=np.zeros((0,) + action_shape),
            rewards=np.zeros((0, 1)),
        )

    def append(
        self, observation: np.ndarray, action: np.ndarray, reward: float
    ) -> "Trajectory":
        """Append a new observation, action, and reward to the trajectory."""
        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        return Trajectory(
            observations=np.concatenate((self.observations, observation[None])),
            actions=np.concatenate((self.actions, action[None])),
            rewards=np.concatenate((self.rewards, np.array([reward])[None])),
        )

SampleActions = Callable[[np.ndarray, Trajectory], np.ndarray]


def evaluate(
    sample_actions: SampleActions,
    env: gym.Env,
    num_episodes: int,
    observation_space: Tuple[int, ...],
    action_space: Tuple[int, ...],
    disable_tqdm: bool = True,
    verbose: bool = False,
    all_results: bool = False,
    num_videos: int = 3,
) -> Dict[str, Any]:
    # stats: Dict[str, Any] = {"return": [], "length": []}
    stats: Dict[str, Any] = defaultdict(list)
    successes = None

    pbar = tqdm.tqdm(range(num_episodes), disable=disable_tqdm, ncols=85)
    videos = []

    for i in pbar:
        observation, done = env.reset(), False
        trajectory_history = Trajectory.create_empty(observation_space, action_space)
        if len(videos) < num_videos:
            try:
                imgs = [env.sim.render(64, 48, camera_name="track")[::-1]]
            except:
                imgs = [env.render()[::-1]]

        i = 0
        while i < 100:
            ob = np.array([])
            for k in observation.keys():
                ob = np.concatenate((ob, observation[k]))
            # print(o)
            action = sample_actions(ob, trajectory_history)
            action = np.clip(action, -1, 1)
            # print("ac", action)
            new_observation, reward, done, info = env.step(action)
            # print("ob", new_observation)
            # print("reward", reward)
            # print("done", done)
            trajectory_history = trajectory_history.append(ob, action, reward)
            observation = new_observation
            if len(videos) < num_videos:
                try:
                    imgs.append(env.sim.render(64, 48, camera_name="track")[::-1])
                except:
                    imgs.append(env.render()[::-1])
            i = i + 1

        if len(videos) < num_videos:
            videos.append(np.array(imgs[:-1]))

        if "episode" in info:
            for k in stats.keys():
                stats[k].append(float(info["episode"][k]))
                if verbose:
                    print(f"{k}: {info['episode'][k]}")

            ret = info["episode"]["return"]
            mean = np.mean(stats["return"])
            pbar.set_description(f"iter={i}\t last={ret:.2f} mean={mean}")
            if "is_success" in info:
                if successes is None:
                    successes = 0.0
                successes += info["is_success"]
        else:
            # breakpoint()
            stats["return"].append(trajectory_history.rewards.sum())
            stats["length"].append(len(trajectory_history.rewards))
            # stats["achieved"].append(int(info["goal_achieved"]))

    new_stats = {}
    for k, v in stats.items():
        new_stats[k + "_mean"] = float(np.mean(v))
        new_stats[k + "_std"] = float(np.std(v))
    if all_results:
        new_stats.update(stats)
    stats = new_stats

    if successes is not None:
        stats["success"] = successes / num_episodes

    return stats, videos

def get_datasets(
    seq_steps: bool,
    noise: float,
    train_dataset_size: int,
    val_dataset_size: int,
    discount: int = 1.5,
    train_val_split: float = 0.95,
    traj_length : int = 16
):
    observations, actions, rewards, states, dones, next_observations, env= load_data_from_hdf5('/home/jonchen25/testing/mtm/research/mtm/datasets/data/liftph.hdf5', 10, noise)
    trajs = split_into_trajectories(observations, actions, rewards, states, dones, next_observations)
    train_size = int(train_val_split * len(trajs))

    (
        train_observations,
        train_actions,
        train_rewards,
        train_masks,
        train_dones_float,
        train_next_observations,
    ) = merge_trajectories(trajs[:train_size])

    (
        valid_observations,
        valid_actions,
        valid_rewards,
        valid_masks,
        valid_dones_float,
        valid_next_observations,
    ) = merge_trajectories(trajs[train_size:])

    train_dataset = RobomimicDataset(
        train_observations,
        train_actions,
        train_rewards,
        train_masks,
        train_dones_float,
        train_next_observations,
        env = env,
        size=len(train_observations),
    )

    valid_dataset = RobomimicDataset(
        valid_observations,
        valid_actions,
        valid_rewards,
        valid_masks,
        valid_dones_float,
        valid_next_observations,
        env = env,
        size=len(valid_observations),
    )

    training_seq = RobomimicSequenceDataset(dataset=train_dataset)
    valid_seq = RobomimicSequenceDataset(dataset=valid_dataset)

    # train_dataset = RobomimicDataset(seq_steps, noise, train_dataset_size)
    # val_dataset = RobomimicDataset(seq_steps, noise, val_dataset_size)
    return training_seq, valid_seq


def load_data_from_hdf5(file_path: str, traj_length: int, noise: float):
    # Open the HDF5 file and extract necessary data
    observations = []
    actions = []
    rewards = []
    states = []
    dones = []
    next_observations = []
    env = None
    with h5py.File(file_path, 'r') as file:
        env_meta = json.loads(file["data"].attrs["env_args"]) 
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta['env_name'], 
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )

        spec = dict(
            obs=dict(
                low_dim=['object','robot0_joint_pos','robot0_joint_pos_cos', 'robot0_joint_pos_sin','robot0_joint_vel',
                         'robot0_eef_pos','robot0_eef_quat','robot0_eef_vel_lin', 'robot0_eef_vel_ang', 'robot0_gripper_qpos',
                          'robot0_gripper_qvel'],
                rgb=[],
            ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=spec)

        for demo in file['data']:
            actions += file['data'][demo]['actions']
            dones += file['data'][demo]['dones']
            rewards += file['data'][demo]['rewards']
            states += file['data'][demo]['states']

            first = True
            traj_observation = None
            traj_observation_next = None
            for ob_i in file['data'][demo]['obs']:
                # print(ob_i)
                if first :
                    traj_observation = np.array( file['data'][demo]['obs'][ob_i])
                    traj_observation_next = np.array( file['data'][demo]['next_obs'][ob_i])
                    first = False
                else:
                    traj_observation = np.concatenate((traj_observation, np.array(file['data'][demo]['obs'][ob_i])), axis = 1)
                    traj_observation_next = np.concatenate((traj_observation_next, np.array(file['data'][demo]['next_obs'][ob_i])), axis = 1)
            for i in range(traj_observation.shape[0]):
                observations.append(traj_observation[i])
                next_observations.append(traj_observation_next[i])

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    states = np.array(states)
    dones = np.array(dones)
    next_observations = np.array(next_observations)
    assert observations.shape[0] == actions.shape[0] == rewards.shape[0] == states.shape[0] == dones.shape[0] == next_observations.shape[0]

    return observations, actions, rewards, states, dones, next_observations, env


def split_into_trajectories(
     observations, actions, rewards, states, dones, next_observations
):
    trajs = [[]]
    for i in range(len(observations)):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                states[i],
                dones[i],
                next_observations[i],
            )
        )
        if dones[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    states = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for obs, act, rew, mask, done, next_obs in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            states.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return (
        np.stack(observations),
        np.stack(actions),
        np.stack(rewards),
        np.stack(states),
        np.stack(dones_float),
        np.stack(next_observations),
    )

@torch.inference_mode()
def sample_action_bc(
    observation: np.ndarray,
    traj: Trajectory,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
):
    traj_len = model.max_len

    # make observations and actions
    observations = np.zeros((traj_len, observation_shape[1]))
    actions = np.zeros((traj_len, action_shape[1]))
    rewards = np.zeros((traj_len, 1))
    # returns = np.zeros((traj_len, 1))
    
    observations[0] = observation
    mask = np.zeros(traj_len)
    obs_mask = np.zeros(traj_len)
    obs_mask[0] = 1
    masks = {
        "states": obs_mask,
        "actions": mask,
        "rewards": mask,
        # "returns": mask,
    }
    trajectories = {
        "states": observations,
        "actions": actions,
        "rewards": rewards,
        # "returns": returns,
    }
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)

    predicted = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    # extract_action
    a = decode["actions"][0][0].cpu().numpy()
    return a

SampleActions = Callable[[np.ndarray, Trajectory], np.ndarray]

##Wrapper class 
class RobomimicDataset(Dataset, DatasetProtocol):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        env : None,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size
        self.env = env

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        """Returns the observation, action, and return for the given index."""
        # print(self.observations[idx].shape)
        # dict = {}
        # dict['observation'] = np.array([self.observations[idx]])
        # dict['actions'] = np.array([self.actions[idx]])
        # dict['rewawrds'] = np.array([self.rewards[idx]])
        # print("rew")
        return {}

    def eval_logs(
        self, model: Callable, tokenizer_manager: TokenizerManager
    ) -> Dict[str, Any]:
       return {}

    def hello_world(self):
        print("HELLO_WORLD")



def segment(observations, terminals, max_path_length):
    """
    segment `observations` into trajectories according to `terminals`
    """
    assert len(observations) == len(terminals)
    observation_dim = observations.shape[1]

    trajectories = [[]]
    for obs, term in zip(observations, terminals):
        trajectories[-1].append(obs)
        if term.squeeze():
            trajectories.append([])

    if len(trajectories[-1]) == 0:
        trajectories = trajectories[:-1]

    ## list of arrays because trajectories lengths will be different
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]

    n_trajectories = len(trajectories)
    path_lengths = [len(traj) for traj in trajectories]

    ## pad trajectories to be of equal length
    trajectories_pad = np.zeros(
        (n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype
    )
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=bool)
    for i, traj in enumerate(trajectories):
        path_length = path_lengths[i]
        trajectories_pad[i, :path_length] = traj
        early_termination[i, path_length:] = 1

    return trajectories_pad, early_termination, path_lengths

class RobomimicSequenceDataset:

    def __init__(
                self,
        dataset: RobomimicDataset,
        discount: float = 0.99,
        sequence_length: int = 4,
        max_path_length: int = 1000,
        use_reward: bool = True,
        name: str = "",
    ):
        self.env = dataset.env
        assert(self.env != None)
        self.dataset = dataset
        self.max_path_length = max_path_length
        self.sequence_length = sequence_length
        self._use_reward = use_reward
        self._name = name

        self.observations_raw = dataset.observations
        self.actions_raw = dataset.actions
        self.rewards_raw = dataset.rewards.reshape(-1, 1)
        self.terminals_raw = dataset.dones_float

        self.actions_segmented, self.termination_flags, self.path_lengths = segment(
            self.actions_raw, self.terminals_raw, max_path_length
        )
        self.observations_segmented, *_ = segment(
            self.observations_raw, self.terminals_raw, max_path_length
        )
        self.rewards_segmented, *_ = segment(
            self.rewards_raw, self.terminals_raw, max_path_length
        )


        if discount > 1.0:
            self.discount = 1.0
            self.use_avg = True
        else:
            self.discount = discount
            self.use_avg = False

        self.discounts = (self.discount ** np.arange(self.max_path_length))[:, None]
        self.values_segmented = np.zeros(self.rewards_segmented.shape)

        for t in range(max_path_length):
            ## [ n_paths x 1 ]
            V = (self.rewards_segmented[:, t + 1 :] * self.discounts[: -t - 1]).sum(
                axis=1
            )
            self.values_segmented[:, t] = V
        
        N_p, Max_Path_Len, _ = self.values_segmented.shape
        if self.use_avg:
            divisor = np.arange(1, Max_Path_Len + 1)[::-1][None, :, None]
            self.values_segmented = self.values_segmented / divisor

        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.termination_flags.reshape(-1)
        self.values_raw = values_raw[values_mask, None]

        self.observation_dim = self.observations_raw.shape[1]
        self.action_dim = self.actions_raw.shape[1]

        assert (
            self.observations_segmented.shape[0]
            == self.actions_segmented.shape[0]
            == self.rewards_segmented.shape[0]
            == self.values_segmented.shape[0]
        )

        keep_idx = []
        index_map = {}
        count = 0
        traj_count = 0
        for idx, pl in enumerate(self.path_lengths):
            if pl < sequence_length:
                pass
            else:
                keep_idx.append(idx)
                for i in range(pl - sequence_length + 1):
                    index_map[count] = (traj_count, i)
                    count += 1
                traj_count += 1

        self.index_map = index_map
        self.path_lengths = np.array(self.path_lengths)[keep_idx]
        self.observations_segmented = self.observations_segmented[keep_idx]
        self.actions_segmented = self.actions_segmented[keep_idx]
        self.rewards_segmented = self.rewards_segmented[keep_idx]
        self.values_segmented = self.values_segmented[keep_idx]
        self.num_trajectories = self.observations_segmented.shape[0]

        self.raw_data = {
            "states": self.observations_raw,
            "actions": self.actions_raw,
            "rewards": self.rewards_raw,
            # "returns": self.values_raw,
        }

    def __len__(self) -> int:
        # return self.num_trajectories
        return len(self.index_map)
    
    @property
    def num_traj(self) -> int:
        return len(self.path_lengths)

    def get_trajectory(self, traj_index: int) -> Dict[str, np.ndarray]:
        if self._use_reward:
            return {
                "states": self.observations_segmented[traj_index],
                "actions": self.actions_segmented[traj_index],
                "rewards": self.rewards_segmented[traj_index],
                # "returns": self.values_segmented[traj_index],
            }
        else:
            return {
                "states": self.observations_segmented[traj_index],
                "actions": self.actions_segmented[traj_index],
            }

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Return a trajectories of the form (observations, actions, rewards, values).

        A random trajectory with self.sequence_length is returned.
        """
        idx, start_idx = self.index_map[index]
        traj = self.get_trajectory(idx)
        return {
            k: v[start_idx : start_idx + self.sequence_length] for k, v in traj.items()
        }

    def eval_logs(
        self, model: Callable, tokenizer_manager: TokenizerManager
    ) -> Dict[str, Any]:
        log_data = {}
        observation_shape = self.observations_raw.shape
        action_shape = self.actions_raw.shape
        device = next(model.parameters()).device


        print("START_EVAL")
        bc_sampler = lambda o, t: sample_action_bc(
            o, t, model, tokenizer_manager, observation_shape, action_shape, device
        )

        results, videos = evaluate(
            bc_sampler,
            self.dataset.env,
            20,
            (self.observation_dim,),
            (self.action_dim,),
            num_videos=0,
        )

        for k, v in results.items():
            log_data[f"eval_bc/{k}"] = v
        for idx, v in enumerate(videos):
            log_data[f"eval_bc_video_{idx}/video"] = wandb.Video(
                v.transpose(0, 3, 1, 2), fps=10, format="gif"
            )

        print("DONE EVAL")
        return log_data


    def trajectory_statistics(self) -> Dict[str, DataStatistics]:
        """Shapes of the trajectories in the dataset."""

        trajectories = {
            "states": self.observations_segmented,
            "actions": self.actions_segmented,
            "rewards": self.rewards_segmented,
            "returns": self.values_segmented,
        }

        # average over samples and time
        ret_dict = {
            k: DataStatistics(
                mean=v.mean(axis=(0, 1)),
                std=v.std(axis=(0, 1)),
                min=v.min(axis=(0, 1)),
                max=v.max(axis=(0, 1)),
            )
            for k, v in trajectories.items()
        }

        return ret_dict