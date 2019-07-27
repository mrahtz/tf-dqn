import argparse
import time
from functools import partial

from dqn.env import make_env
from dqn.model import Model
from dqn.policies import CNNFeatures, Policy

parser = argparse.ArgumentParser()
parser.add_argument('env_id')
parser.add_argument('ckpt_dir')
args = parser.parse_args()

env = make_env(args.env_id, seed=0)

policy_fn = partial(Policy, features_cls=CNNFeatures, dueling=True)
model = Model(policy_fn=policy_fn, obs_shape=env.observation_space.shape, n_actions=env.action_space.n,
              discount=0.99, lr=1e-5, gradient_clip=10, seed=0, double=True)
model.load(args.ckpt_dir)

while True:
    done, obs = False, env.reset()
    while not done:
        action = model.step(obs, random_action_prob=0)
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1/60)
