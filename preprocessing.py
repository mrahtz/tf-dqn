# From https://github.com/mrahtz/ocd-a3c/blob/master/preprocessing.py

from collections import deque

import cv2
import numpy as np
from gym import spaces
from gym.core import Wrapper, ObservationWrapper, RewardWrapper
from gym.spaces import Box


def get_noop_action_index(env):
    action_meanings = env.unwrapped.get_action_meanings()
    try:
        noop_action_index = action_meanings.index('NOOP')
        return noop_action_index
    except ValueError:
        raise Exception("Unsure about environment's no-op action")


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
        reward_sum = 0
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


"""
We also have a wrapper to extract hand-crafted features from Pong for early 
debug testing.
"""


class PongFeaturesWrapper(ObservationWrapper):
    """
    Manually extract the Pong game area, setting paddles/ball to 1.0 and the background to 0.0.
    """

    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(84, 84), dtype=np.float32)

    def observation(self, obs):
        """
        Based on Andrej Karpathy's code for Pong with policy gradients:
        https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
        """
        obs = np.mean(obs, axis=2) / 255.0  # Convert to [0, 1] grayscale
        obs = obs[34:194]  # Extract game area
        obs = obs[::2, ::2]  # Downsample by a factor of 2
        obs = np.pad(obs, pad_width=2, mode='constant')  # Pad to 84x84
        obs[obs <= 0.4] = 0  # Erase background
        obs[obs > 0.4] = 1  # Set balls, paddles to 1
        return obs

def pong_preprocess(env, max_n_noops):
    env = RandomStartWrapper(env, max_n_noops)
    env = PongFeaturesWrapper(env)
    env = FrameSkipWrapper(env)
    env = FrameStackWrapper(env)
    return env
