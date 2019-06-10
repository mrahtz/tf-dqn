import multiprocessing
import os

import cloudpickle as cloudpickle
import easy_tf_log
import gym
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.stflow import LogFileWriter

from model import Model
from utils import tf_disable_warnings, tf_disable_deprecation_warnings


class ExperienceBatch:
    def __init__(self, o1s, acts, rs, o2s, dones):
        self.o1s = o1s
        self.acts = acts
        self.rs = rs
        self.o2s = o2s
        self.dones = dones


class ExperienceBuffer:
    def __init__(self, obs_shape, max_size=100_000):
        self.o1s = np.zeros((max_size,) + obs_shape)
        self.o2s = np.zeros((max_size,) + obs_shape)
        self.rs = np.zeros(max_size)
        self.acts = np.zeros(max_size)
        self.dones = np.zeros(max_size)
        self.max_size = max_size
        self.len = 0
        self.idx = 0

    def __len__(self):
        return self.len

    def store(self, o1, a, r, o2, done):
        self.o1s[self.idx] = o1
        self.acts[self.idx] = a
        self.rs[self.idx] = r
        self.o2s[self.idx] = o2
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.max_size
        if self.len < self.max_size:
            self.len += 1

    def sample(self, batch_size=32) -> ExperienceBatch:
        idxs = np.random.choice(self.len, size=batch_size, replace=False)
        return ExperienceBatch(
            o1s=[self.o1s[i] for i in idxs],
            acts=[self.acts[i] for i in idxs],
            rs=[self.rs[i] for i in idxs],
            o2s=[self.o2s[i] for i in idxs],
            dones=[self.dones[i] for i in idxs]
        )


tf_disable_warnings()
tf_disable_deprecation_warnings()

ex = Experiment('dqn')
observer = FileStorageObserver.create('runs')
ex.observers.append(observer)


@ex.config
def cfg():
    batch_size = 32
    gamma = 1.0
    eps = 0.9
    n_start_steps = 1000
    update_target_every_n_steps = 500
    log_every_n_steps = 100
    checkpoint_every_n_steps = 100
    buffer_size = 50000
    lr = 5e-4
    render = False


@ex.named_config
def render():
    render = True


def get_random_action_prob(step_n):
    return 1 - (1 - 0.02) * min(1, step_n / (1e5 * 0.1))


@ex.capture
@LogFileWriter(ex)
def train(experience_buffer: ExperienceBuffer,
          model: Model,
          train_env,
          batch_size,
          n_start_steps,
          update_target_every_n_steps,
          log_every_n_steps,
          checkpoint_every_n_steps):
    logger = easy_tf_log.Logger()
    logger.set_log_dir(observer.dir)
    step_n = 0
    episode_reward = 0
    episode_rewards = []
    obs1, done = train_env.reset(), False

    while True:
        random_action_prob = get_random_action_prob(step_n)

        action = model.step(obs1, random_action_prob=random_action_prob)
        obs2, reward, done, info = train_env.step(action)
        episode_reward += reward
        experience_buffer.store(o1=obs1, a=action, r=reward, o2=obs2, done=float(done))
        obs1 = obs2
        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            obs1, done = train_env.reset(), False

        if len(experience_buffer) < n_start_steps:
            continue

        experience_batch = experience_buffer.sample(batch_size=batch_size)
        model.train(o1s=experience_batch.o1s,
                    acts=experience_batch.acts,
                    rs=experience_batch.rs,
                    o2s=experience_batch.o2s,
                    dones=experience_batch.dones)

        if step_n % update_target_every_n_steps == 0:
            model.update_target()

        if step_n % log_every_n_steps == 0:
            log_stats(random_action_prob, step_n, episode_rewards, experience_buffer)

        if step_n % checkpoint_every_n_steps == 0:
            model.save()

        step_n += 1


def log_stats(random_action_prob, step_n, train_episode_rewards, buffer):
    print(f"Step: {step_n}")
    print("Random action probability: {:.2f}".format(random_action_prob))
    print("Last 10 train episode mean reward sum:", np.mean(train_episode_rewards[-10:]))
    print("Buffer size:", len(buffer))
    print()


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
def main(gamma, buffer_size, lr, render):
    train_env = gym.make('CartPole-v0')
    experience_buffer = ExperienceBuffer(train_env.observation_space.shape, max_size=buffer_size)
    make_model_fn = lambda **kwargs: Model(train_env.observation_space.shape,
                                           train_env.action_space.n,
                                           discount=gamma,
                                           lr=lr,
                                           **kwargs)
    ckpt_dir = os.path.join(observer.dir, 'checkpoints')
    model = make_model_fn(save_dir=ckpt_dir)
    model.save()
    if render:
        ctx = multiprocessing.get_context('spawn')
        ctx.Process(target=run_test_env, args=(cloudpickle.dumps(make_model_fn), ckpt_dir)).start()
    train(experience_buffer, model, train_env)
