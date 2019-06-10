import multiprocessing
import os

import cloudpickle as cloudpickle
import gym
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.stflow import LogFileWriter

from model import Model
from replay_buffer import ReplayBuffer
from utils import tf_disable_warnings, tf_disable_deprecation_warnings

tf_disable_warnings()
tf_disable_deprecation_warnings()

ex = Experiment('dqn')
observer = FileStorageObserver.create('runs')
ex.observers.append(observer)


@ex.config
def cfg():
    batch_size = 32
    gamma = 1.0
    random_action_prob = 0.1
    n_start_steps = 1000
    update_target_every_n_steps = 500
    log_every_n_steps = 100
    checkpoint_every_n_steps = 100
    buffer_size = 50000
    lr = 5e-4
    render = False
    n_hidden = 64


@ex.named_config
def render():
    render = True


@ex.capture
@LogFileWriter(ex)
def train(replay_buffer: ReplayBuffer,
          model: Model,
          train_env,
          batch_size,
          n_start_steps,
          update_target_every_n_steps,
          log_every_n_steps,
          checkpoint_every_n_steps,
          random_action_prob):
    step_n = 0
    episode_reward = 0
    episode_rewards = []
    obs1, done = train_env.reset(), False

    while True:
        act = model.step(obs1, random_action_prob=random_action_prob)
        obs2, reward, done, info = train_env.step(act)
        replay_buffer.store(obs1=obs1, acts=act, rews=reward, obs2=obs2, done=float(done))
        episode_reward += reward
        obs1 = obs2
        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            obs1, done = train_env.reset(), False

        if len(replay_buffer) < n_start_steps:
            continue

        batch = replay_buffer.sample(batch_size=batch_size)
        model.train(batch)

        if step_n % update_target_every_n_steps == 0:
            model.update_target()

        if step_n % log_every_n_steps == 0:
            print(f"Step: {step_n}")
            print("Last 10 train episode mean reward sum:", np.mean(episode_rewards[-10:]))
            print("Buffer size:", len(replay_buffer))

        if step_n % checkpoint_every_n_steps == 0:
            model.save()

        step_n += 1


def run_test_env(make_model_fn_pkl, model_load_dir):
    model = cloudpickle.loads(make_model_fn_pkl)()
    test_env = gym.make('CartPole-v0')
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
def main(gamma, buffer_size, lr, render, seed, n_hidden):
    train_env = gym.make('CartPole-v0')
    experience_buffer = ReplayBuffer(train_env.observation_space.shape, max_size=buffer_size)
    make_model_fn = lambda **kwargs: Model(train_env.observation_space.shape,
                                           train_env.action_space.n,
                                           discount=gamma,
                                           lr=lr,
                                           seed=seed,
                                           n_hidden=n_hidden,
                                           **kwargs)
    ckpt_dir = os.path.join(observer.dir, 'checkpoints')
    model = make_model_fn(save_dir=ckpt_dir)
    model.save()
    if render:
        ctx = multiprocessing.get_context('spawn')
        ctx.Process(target=run_test_env, args=(cloudpickle.dumps(make_model_fn), ckpt_dir)).start()
    train(experience_buffer, model, train_env)
