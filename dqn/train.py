import multiprocessing
import os
import sys
from functools import partial

import easy_tf_log
import gym
import numpy as np
from gym.envs.atari import AtariEnv
from sacred import Experiment
from sacred.observers import FileStorageObserver

import dqn.config as config
from dqn.env import make_env
from dqn.model import Model
from dqn.policies import MLPFeatures, CNNFeatures, Policy
from dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from dqn.utils import tf_disable_warnings, tf_disable_deprecation_warnings, RateMeasure

tf_disable_warnings()
tf_disable_deprecation_warnings()

ex = Experiment('dqn')
observer = FileStorageObserver.create('runs')
ex.observers.append(observer)
ex.add_config(config.default_config)
ex.add_named_config('atari_config', config.atari_config)


@ex.capture
def train_dqn(buffer: ReplayBuffer, model: Model, env,
              batch_size, n_start_steps, update_target_every_n_steps, log_every_n_steps, checkpoint_every_n_steps,
              random_action_prob_anneal_frac, random_action_prob_final, train_n_steps, n_env_steps_per_rl_update):
    obs1, done = env.reset(), False
    logger = easy_tf_log.Logger(os.path.join(observer.dir))
    n_steps, episode_reward = 0, 0
    step_rate_measure = RateMeasure(n_steps)
    losses = []

    while n_steps < train_n_steps:
        anneal_final_step = train_n_steps * random_action_prob_anneal_frac
        anneal_frac = np.minimum(1, n_steps / anneal_final_step)
        random_action_prob = 1 - anneal_frac * (1 - random_action_prob_final)

        act = model.step(obs1, random_action_prob)
        obs2, reward, done, info = env.step(act)
        try:
            episode_reward += info['unclipped_reward']
        except KeyError:
            episode_reward += reward
        if isinstance(buffer, PrioritizedReplayBuffer):
            td = model.calculate_td(obs1=obs1, act=act, reward=reward, obs2=obs2, done=done)
            buffer.store_with_td(obs1=obs1, acts=act, rews=reward, obs2=obs2, done=float(done), td=td)
            buffer.update_beta(n_steps / train_n_steps)
        else:
            buffer.store(obs1=obs1, acts=act, rews=reward, obs2=obs2, done=float(done))
        obs1 = obs2
        if done:
            logger.logkv('env_train/episode_reward', episode_reward)
            episode_reward = 0
            obs1, done = env.reset(), False

        if len(buffer) < n_start_steps:
            continue

        if n_steps % n_env_steps_per_rl_update == 0:
            batch = buffer.sample(batch_size=batch_size)
            loss, td_errors = model.train(batch)
            losses.append(loss)
            if isinstance(buffer, PrioritizedReplayBuffer):
                buffer.update_tds(batch.idxs, td_errors)

        if n_steps % update_target_every_n_steps == 0:
            model.update_target()

        if n_steps % log_every_n_steps == 0:
            print(f"Trained {n_steps} steps")
            n_steps_per_second = step_rate_measure.measure(n_steps)
            logger.logkv('dqn/buffer_size', len(buffer))
            logger.logkv('dqn/n_steps', n_steps)
            logger.logkv('dqn/n_steps_per_second', n_steps_per_second)
            logger.logkv('dqn/loss', np.mean(losses))
            logger.logkv('dqn/random_action_prob', random_action_prob)
            if isinstance(buffer, PrioritizedReplayBuffer):
                logger.logkv('replay_buffer/beta', buffer.beta)
            losses = []

        if n_steps % checkpoint_every_n_steps == 0:
            model.save()

        n_steps += 1


def async_test_episodes_loop(model, model_load_dir, env_id, seed, log_dir):
    env = gym.make(env_id)
    env.seed(seed)
    logger = easy_tf_log.Logger(os.path.join(log_dir, 'test_loop'))
    while True:
        model.load(model_load_dir)
        run_test_episodes(model, env, n_episodes=1, logger=logger, render=True)


def run_test_episodes(model, env, n_episodes, logger, render):
    episodes_reward = []
    for _ in range(n_episodes):
        obs, done, = env.reset(), False
        episode_reward = 0
        while not done:
            action = model.step(obs, random_action_prob=0)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
        episodes_reward.append(episode_reward)
    mean_reward = np.mean(episodes_reward)
    print("Test episodes done; mean reward {:.1f}".format(mean_reward))
    sys.stdout.flush()
    logger.logkv('env_test/episode_reward', mean_reward)


@ex.automain
def main(gamma, buffer_size, lr, gradient_clip, seed, env_id, double, dueling, prioritized,
         features, async_test):
    env = make_env(env_id, seed)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape

    if isinstance(env.unwrapped, AtariEnv) and features != 'cnn':
        raise Exception("Atari environments must use atari_config")

    if features == 'mlp':
        features_cls = MLPFeatures
    elif features == 'cnn':
        features_cls = CNNFeatures
    else:
        raise Exception(f"Invalid features network: '{features}'")
    policy_fn = partial(Policy, features_cls=features_cls, dueling=dueling)

    if prioritized:
        buffer = PrioritizedReplayBuffer(env.observation_space.shape, max_size=buffer_size)
    else:
        buffer = ReplayBuffer(env.observation_space.shape, max_size=buffer_size)
    ckpt_dir = os.path.join(observer.dir, 'checkpoints')
    model = Model(policy_fn=policy_fn, obs_shape=obs_shape, n_actions=n_actions, save_dir=ckpt_dir,
                  discount=gamma, lr=lr, gradient_clip=gradient_clip, seed=seed, double=double)
    model.save()

    if async_test:
        ctx = multiprocessing.get_context('spawn')
        test_env_proc = ctx.Process(target=async_test_episodes_loop, daemon=True,
                                    args=(model, ckpt_dir, env_id, seed, observer.dir))
        test_env_proc.start()
    else:
        test_env_proc = None

    train_dqn(buffer, model, env)

    if test_env_proc:
        test_env_proc.terminate()
    return model
