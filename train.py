import multiprocessing
import os
from functools import partial

import easy_tf_log
import numpy as np
from gym.envs.atari import AtariEnv
from sacred import Experiment
from sacred.observers import FileStorageObserver

import config
from env import make_env
from model import Model
from policies import cnn_features, mlp_features, make_policy
from replay_buffer import ReplayBuffer
from utils import tf_disable_warnings, tf_disable_deprecation_warnings, RateMeasure

tf_disable_warnings()
tf_disable_deprecation_warnings()

ex = Experiment('dqn')
observer = FileStorageObserver.create('runs')
ex.observers.append(observer)
ex.add_config(config.default_config)
ex.add_named_config('atari_config', config.atari_config)


@ex.capture
def train_dqn(buffer: ReplayBuffer,
              model: Model,
              train_env,
              batch_size,
              n_start_steps,
              update_target_every_n_steps,
              log_every_n_steps,
              checkpoint_every_n_steps,
              random_action_prob,
              train_n_steps,
              n_env_steps_per_rl_update):
    n_steps = 0
    obs1, done = train_env.reset(), False
    logger = easy_tf_log.Logger(os.path.join(observer.dir, 'dqn'))
    step_rate_measure = RateMeasure(n_steps)
    losses = []

    while n_steps < train_n_steps:
        act = model.step(obs1, random_action_prob=random_action_prob)
        obs2, reward, done, info = train_env.step(act)
        buffer.store(obs1=obs1, acts=act, rews=reward, obs2=obs2, done=float(done))
        obs1 = obs2
        if done:
            obs1, done = train_env.reset(), False

        if len(buffer) < n_start_steps:
            continue

        if n_steps % n_env_steps_per_rl_update == 0:
            batch = buffer.sample(batch_size=batch_size)
            loss = model.train(batch)
            losses.append(loss)

        if n_steps % update_target_every_n_steps == 0:
            model.update_target()

        if n_steps % checkpoint_every_n_steps == 0:
            model.save()

        if n_steps % log_every_n_steps == 0:
            print(f"Trained {n_steps} steps")
            n_steps_per_second = step_rate_measure.measure(n_steps)
            logger.logkv('dqn/buffer_size', len(buffer))
            logger.logkv('dqn/n_steps', n_steps)
            logger.logkv('dqn/n_steps_per_second', n_steps_per_second)
            logger.logkv('dqn/loss', np.mean(losses))
            losses = []

        n_steps += 1


def run_test_env(model, model_load_dir, render, env_id, seed, log_dir):
    env = make_env(env_id, seed, log_dir, 'test')
    while True:
        model.load(model_load_dir)
        obs, done, rewards = env.reset(), False, []
        while not done:
            action = model.step(obs, random_action_prob=0)
            if render:
                env.render()
            obs, reward, done, info = env.step(action)


@ex.automain
def main(gamma, buffer_size, lr, render, seed, env_id, double_dqn, dueling, feature_extractor=None):
    env = make_env(env_id, seed, observer.dir, 'train')

    if feature_extractor is None:
        if isinstance(env.unwrapped, AtariEnv):
            feature_extractor = cnn_features
        else:
            feature_extractor = mlp_features
    policy_fn = partial(make_policy, feature_extractor=feature_extractor, dueling=dueling)

    buffer = ReplayBuffer(env.observation_space.shape, max_size=buffer_size)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    ckpt_dir = os.path.join(observer.dir, 'checkpoints')
    model = Model(policy_fn=policy_fn, obs_shape=obs_shape, n_actions=n_actions, save_dir=ckpt_dir,
                  discount=gamma, lr=lr, seed=seed, double_dqn=double_dqn)
    model.save()

    ctx = multiprocessing.get_context('spawn')
    test_env_proc = ctx.Process(target=run_test_env, daemon=False,
                                args=(model, ckpt_dir, render, env_id, seed, observer.dir))
    test_env_proc.start()

    train_dqn(buffer, model, env)

    test_env_proc.terminate()
    return model
