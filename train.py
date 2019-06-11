import multiprocessing
import os
import pickle

import cloudpickle as cloudpickle
import easy_tf_log
import gym
import numpy as np
from gym.envs.atari import AtariEnv
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.stflow import LogFileWriter

import config
from model import Model
from policies import MLPPolicy, CNNPolicy
from preprocessing import atari_preprocess
from replay_buffer import ReplayBuffer
from utils import tf_disable_warnings, tf_disable_deprecation_warnings

tf_disable_warnings()
tf_disable_deprecation_warnings()

ex = Experiment('dqn')
observer = FileStorageObserver.create('runs')
ex.observers.append(observer)
ex.add_config(config.params)


@ex.capture
@LogFileWriter(ex)
def train_dqn(buffer: ReplayBuffer,
              model: Model,
              train_env,
              batch_size,
              n_start_steps,
              update_target_every_n_steps,
              log_every_n_steps,
              checkpoint_every_n_steps,
              random_action_prob,
              train_n_steps):
    logger = easy_tf_log.Logger(os.path.join(observer.dir, 'train-env'))
    step_n = 0
    episode_reward = 0
    episode_rewards = []
    obs1, done = train_env.reset(), False

    while step_n < train_n_steps:
        act = model.step(obs1, random_action_prob=random_action_prob)
        obs2, reward, done, info = train_env.step(act)
        buffer.store(obs1=obs1, acts=act, rews=reward, obs2=obs2, done=float(done))
        episode_reward += reward
        obs1 = obs2
        if done:
            logger.logkv('train_reward', episode_reward)
            episode_rewards.append(episode_reward)
            episode_reward = 0
            obs1, done = train_env.reset(), False

        if len(buffer) < n_start_steps:
            continue

        batch = buffer.sample(batch_size=batch_size)
        model.train(batch)

        if step_n % update_target_every_n_steps == 0:
            model.update_target()

        if step_n % log_every_n_steps == 0:
            print(f"Step: {step_n}")
            print("Last 10 train episode mean reward sum:", np.mean(episode_rewards[-10:]))
            print("Buffer size:", len(buffer))

        if step_n % checkpoint_every_n_steps == 0:
            model.save()

        step_n += 1

def run_test_env(make_model_fn_pkl, model_load_dir, render, log_dir, env):
    model = cloudpickle.loads(make_model_fn_pkl)()
    logger = easy_tf_log.Logger(log_dir)
    while True:
        model.load(model_load_dir)
        obs, done = env.reset(), False
        rewards = []
        while not done:
            action = model.step(obs, random_action_prob=0)
            if render:
                env.render()
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
        episode_reward = sum(rewards)
        logger.logkv('test_reward', episode_reward)
        print("Test episode reward:", episode_reward)


@ex.automain
def main(gamma, buffer_size, lr, render, seed, env_id, double_dqn):
    env = gym.make(env_id)
    env.seed(seed)
    if isinstance(env.unwrapped, AtariEnv):
        env = atari_preprocess(env)
        policy = CNNPolicy
    else:
        policy = MLPPolicy
    buffer = ReplayBuffer(env.observation_space.shape, max_size=buffer_size)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    make_model_fn = lambda **kwargs: Model(policy, obs_shape, n_actions,
                                           discount=gamma, lr=lr, seed=seed, double_dqn=double_dqn,
                                           **kwargs)
    ckpt_dir = os.path.join(observer.dir, 'checkpoints')
    model = make_model_fn(save_dir=ckpt_dir)
    model.save()
    ctx = multiprocessing.get_context('spawn')
    ctx.Process(target=run_test_env,
                args=(cloudpickle.dumps(make_model_fn), ckpt_dir, render, os.path.join(observer.dir, 'test_env'), env)).start()
    train_dqn(buffer, model, env)

    return model
