import gym
import numpy as np

from model import Model


class ExperienceBatch:
    def __init__(self, o1s, acts, rs, o2s, dones):
        self.o1s = o1s
        self.acts = acts
        self.rs = rs
        self.o2s = o2s
        self.dones = dones


class ExperienceBuffer:
    def __init__(self, obs_shape, max_size=100000):
        self.max_size = max_size
        self.o1s = np.zeros((max_size,) + obs_shape)
        self.o2s = np.zeros((max_size,) + obs_shape)
        self.rs = np.zeros(max_size)
        self.acts = np.zeros(max_size)
        self.dones = np.zeros(max_size)
        self.len = 0
        self.idx = 0

    def __len__(self):
        return self.len

    def store(self, o1, a, r, o2, done):
        self.o1s[self.idx] = o1
        self.o2s[self.idx] = o2
        self.rs[self.idx] = r
        self.acts[self.idx] = a
        # TODO float vs bool
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.max_size
        self.len = min(self.len + 1, self.max_size)

    def sample(self, batch_size=256) -> ExperienceBatch:
        idxs = np.random.choice(self.len, size=batch_size, replace=False)
        return ExperienceBatch(
            o1s=[self.o1s[i] for i in idxs],
            acts=[self.acts[i] for i in idxs],
            rs=[self.rs[i] for i in idxs],
            o2s=[self.o2s[i] for i in idxs],
            dones=[self.dones[i] for i in idxs]
        )


def main():
    env = gym.make('CartPole-v0')
    experience_buffer = ExperienceBuffer(env.observation_space.shape)
    model = Model(env.observation_space.shape, env.action_space.n)

    n = 0
    while True:
        for _ in range(10):
            obs1, done = env.reset(), False
            while not done:
                action = model.step(obs1, deterministic=False)
                obs2, reward, done, info = env.step(action)
                experience_buffer.store(obs1, action, reward, obs2, done)
                obs1 = obs2
            n += 1
            print(f"Train episode {n} done")

        if len(experience_buffer) < 256:
            continue

        for _ in range(100):
            experience_batch = experience_buffer.sample()
            loss = model.train(o1s=experience_batch.o1s,
                        acts=experience_batch.acts,
                        rs=experience_batch.rs,
                        o2s=experience_batch.o2s,
                        dones=experience_batch.dones)
            print("Loss: {}".format(loss))

        o, done = env.reset(), False
        rewards = []
        while not done:
            action = model.step(o, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            rewards.append(reward)
        print(sum(rewards))


if __name__ == '__main__':
    main()
