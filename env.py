import os
from collections import deque

import cv2
import easy_tf_log
import gym
import numpy as np
from gym import spaces
from gym.core import Wrapper, ObservationWrapper, RewardWrapper
from gym.envs.atari import AtariEnv
from gym.spaces import Box


def get_noop_action_index(env):
    action_meanings = env.unwrapped.get_action_meanings()
    try:
        noop_action_index = action_meanings.index('NOOP')
        return noop_action_index
    except ValueError:
        raise Exception("Unsure about environment's no-op action")


# Atari wrappers from https://github.com/mrahtz/ocd-a3c/blob/master/preprocessing.py


class MaxWrapper(Wrapper):
    """
    Take maximum pixel values over pairs of frames.
    """

    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.frame_pairs = deque(maxlen=2)

    def reset(self):
        obs = self.env.reset()
        self.frame_pairs.append(obs)

        # The first frame returned should be the maximum of frames 0 and 1.
        # We get frame 0 from env.reset(). For frame 1, we take a no-op action.
        noop_action_index = get_noop_action_index(self.env)
        obs, _, done, _ = self.env.step(noop_action_index)
        if done:
            raise Exception("Environment signalled done during initial frame maxing")
        self.frame_pairs.append(obs)

        return np.max(self.frame_pairs, axis=0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frame_pairs.append(obs)
        obs_maxed = np.max(self.frame_pairs, axis=0)
        return obs_maxed, reward, done, info


class ExtractLuminanceAndScaleWrapper(ObservationWrapper):
    """
    Convert observations from colour to grayscale, then scale to 84 x 84
    """

    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        # Important so that gym's play.py picks up the right resolution
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Bilinear interpolation
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_LINEAR)
        return obs


class FrameStackWrapper(Wrapper):
    """
    Stack the most recent 4 frames together.
    """

    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.frame_stack = deque(maxlen=4)
        low = np.tile(env.observation_space.low[..., np.newaxis], 4)
        high = np.tile(env.observation_space.high[..., np.newaxis], 4)
        dtype = env.observation_space.dtype
        self.observation_space = Box(low=low, high=high, dtype=dtype)

    def _get_obs(self):
        obs = np.array(self.frame_stack)
        # Switch from (4, 84, 84) to (84, 84, 4), so that we have the right order for inputting
        # directly into the convnet with the default channels_last
        obs = np.moveaxis(obs, 0, -1)
        return obs

    def reset(self):
        obs = self.env.reset()
        self.frame_stack.append(obs)
        # The first observation returned should be a stack of observations 0 through 3. We get
        # observation 0 from env.reset(). For the rest, we take no-op actions.
        noop_action_index = get_noop_action_index(self.env)
        for _ in range(3):
            obs, _, done, _ = self.env.step(noop_action_index)
            if done:
                raise Exception("Environment signalled done during initial "
                                "frame stack")
            self.frame_stack.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frame_stack.append(obs)
        return self._get_obs(), reward, done, info


class FrameSkipWrapper(Wrapper):
    """
    Repeat the chosen action for 4 frames, only returning the last frame.
    """

    def reset(self):
        return self.env.reset()

    def step(self, action):
        reward_sum, obs, done, info = 0, None, None, None
        for _ in range(4):
            obs, reward, done, info = self.env.step(action)
            reward_sum += reward
            if done:
                break
        return obs, reward_sum, done, info


class RandomStartWrapper(Wrapper):
    """
    Start each episode with a random number of no-ops.
    """

    def __init__(self, env, max_n_noops):
        Wrapper.__init__(self, env)
        self.max_n_noops = max_n_noops

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        n_noops = np.random.randint(low=0, high=self.max_n_noops + 1)
        noop_action_index = get_noop_action_index(self.env)
        for _ in range(n_noops):
            obs, _, done, _ = self.env.step(noop_action_index)
            if done:
                raise Exception("Environment signalled done during initial no-ops")
        return obs


class NormalizeObservationsWrapper(ObservationWrapper):
    """
    Normalize observations to range [0, 1].
    """

    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, obs):
        return obs / 255.0


class ClipRewardsWrapper(RewardWrapper):
    """
    Clip rewards to range [-1, +1].
    """

    def reward(self, reward):
        return np.clip(reward, -1, +1)


class EndEpisodeOnLifeLossWrapper(Wrapper):
    """
    Send 'episode done' when life lost. (Baselines' atari_wrappers.py claims that this helps with
    value estimation. I guess it makes it clear that only actions since the last loss of life
    contributed significantly to any rewards in the present.)
    """

    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.done_because_life_lost = False
        self.reset_obs = None

    def step(self, action):
        lives_before = self.env.unwrapped.ale.lives()
        obs, reward, done, info = self.env.step(action)
        lives_after = self.env.unwrapped.ale.lives()

        if done:
            self.done_because_life_lost = False
        elif lives_after < lives_before:
            self.done_because_life_lost = True
            self.reset_obs = obs
            done = True

        return obs, reward, done, info

    def reset(self):
        assert self.done_because_life_lost is not None
        # If we sent the 'episode done' signal after a loss of a life, then we'll probably get a
        # reset signal next. But we shouldn't actually reset! We should just keep on playing until
        # the /real/ end-of-episode.
        if self.done_because_life_lost:
            self.done_because_life_lost = None
            return self.reset_obs
        else:
            return self.env.reset()


def atari_preprocess(env):
    env = RandomStartWrapper(env, max_n_noops=30)
    env = MaxWrapper(env)
    env = ExtractLuminanceAndScaleWrapper(env)
    env = NormalizeObservationsWrapper(env)
    env = FrameSkipWrapper(env)
    env = FrameStackWrapper(env)
    env = EndEpisodeOnLifeLossWrapper(env)
    env = ClipRewardsWrapper(env)
    return env


class LogRewards(Wrapper):
    def __init__(self, env, logger, test_or_train):
        super().__init__(env)
        self.logger = logger
        self.test_or_train = test_or_train
        self.episode_reward = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.episode_reward = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        if done:
            self.logger.logkv(f'env_{self.test_or_train}/episode_reward', self.episode_reward)
        return obs, reward, done, info


def make_env(env_id, seed, log_dir, test_or_train):
    env = gym.make(env_id)
    env.seed(seed)
    logger = easy_tf_log.Logger(os.path.join(log_dir, f'env_{test_or_train}'))
    env = LogRewards(env, logger, test_or_train)
    if isinstance(env.unwrapped, AtariEnv):
        env = atari_preprocess(env)
    return env
