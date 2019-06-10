import multiprocessing
import os
import pickle

import cloudpickle as cloudpickle
import gym
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.stflow import LogFileWriter

import config
from model import Model
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

def run_test_env(make_model_fn_pkl, model_load_dir, env_id):
    model = cloudpickle.loads(make_model_fn_pkl)()
    test_env = gym.make(env_id)
    while True:
        model.load(model_load_dir)
        obs, done = test_env.reset(), False
        rewards = []
        while not done:
            action = model.step(obs, random_action_prob=0)
            test_env.render()
            obs, reward, done, info = test_env.step(action)
            rewards.append(reward)
        print("Test episode reward:", sum(rewards))


@ex.automain
def main(gamma, buffer_size, lr, render, seed, n_hidden, env_id, double_dqn):
    train_env = gym.make(env_id)
    train_env.seed(seed)
    buffer = ReplayBuffer(train_env.observation_space.shape, max_size=buffer_size)
    obs_shape = train_env.observation_space.shape
    n_actions = train_env.action_space.n
    make_model_fn = lambda **kwargs: Model(obs_shape, n_actions,
                                           discount=gamma, lr=lr, n_hidden=n_hidden, seed=seed, double_dqn=double_dqn,
                                           **kwargs)
    ckpt_dir = os.path.join(observer.dir, 'checkpoints')
    model = make_model_fn(save_dir=ckpt_dir)
    model.save()
    if render:
        ctx = multiprocessing.get_context('spawn')
        ctx.Process(target=run_test_env, args=(cloudpickle.dumps(make_model_fn), ckpt_dir, env_id)).start()
    train_dqn(buffer, model, train_env)

    return model
