import numpy as np


class ReplayBatch:
    def __init__(self, obs1, acts, rews, obs2, done, impw=None, idxs=None):
        self.n = len(obs1)
        if impw is None:
            impw = np.ones(self.n)
        self.obs1 = obs1
        self.acts = acts
        self.rews = rews
        self.obs2 = obs2
        self.done = done
        self.impw = impw
        self.idxs = idxs

    def __len__(self):
        return self.n


class ReplayBuffer:
    def __init__(self, obs_shape, max_size):
        obs_shape = list(obs_shape)
        self.obs1_buf = np.zeros([max_size] + obs_shape)
        self.obs2_buf = np.zeros([max_size] + obs_shape)
        self.rews_buf = np.zeros(max_size)
        self.acts_buf = np.zeros(max_size)
        self.done_buf = np.zeros(max_size)
        self.idx = 0
        self.len = 0
        self.max_size = max_size

    def __len__(self):
        return self.len

    def store(self, obs1, acts, rews, obs2, done):
        self.obs1_buf[self.idx] = obs1
        self.acts_buf[self.idx] = acts
        self.rews_buf[self.idx] = rews
        self.obs2_buf[self.idx] = obs2
        self.done_buf[self.idx] = done

        self.idx = (self.idx + 1) % self.max_size
        if self.len < self.max_size:
            self.len += 1

    def sample(self, batch_size=32) -> ReplayBatch:
        idxs = np.random.choice(self.len, size=batch_size, replace=False)
        return ReplayBatch(
            obs1=[self.obs1_buf[i] for i in idxs],
            acts=[self.acts_buf[i] for i in idxs],
            rews=[self.rews_buf[i] for i in idxs],
            obs2=[self.obs2_buf[i] for i in idxs],
            done=[self.done_buf[i] for i in idxs]
        )


class DumbPrioritizedReplayBuffer(ReplayBuffer):
    """
    "Dumb" prioritized replay: when annealing the weights, do it by changing the sampling probabilities
    rather than by re-weighting the samples. (I have no idea why it's not done this way in practice. Importance
    sampling seems so much more complicated.)
    """

    # Hyperparameters from
    # https://github.com/hill-a/stable-baselines/blob/72dab6a37bb33ad959cc598f187299edc49c425f/stable_baselines/deepq/dqn.py
    def __init__(self, obs_shape, max_size, alpha=0.6, beta0=0.4, eps=1e-6):
        super().__init__(obs_shape, max_size)
        self.tds_buf = np.zeros(max_size)
        self.alpha = alpha
        self.beta0 = beta0
        self.beta = beta0
        self.eps = eps

    def store_with_td(self, obs1, acts, rews, obs2, done, td):
        self.tds_buf[self.idx] = td
        self.store(obs1, acts, rews, obs2, done)

    def update_tds(self, idxs, tds):
        for i, td in zip(idxs, tds):
            self.tds_buf[i] = td

    def update_beta(self, cur_frac_of_total_training):
        self.beta = self.beta0 + (1 - self.beta0) * cur_frac_of_total_training

    def sample(self, batch_size=32) -> ReplayBatch:
        tds = self.tds_buf[:self.len]
        priorities = np.abs(tds) + self.eps

        priority_sum = np.sum(priorities)
        assert isinstance(priority_sum, np.float64)
        sample_probs = priorities / priority_sum
        assert sample_probs.shape == (self.len,)

        annealing_weights = (1 / sample_probs) ** self.beta
        assert annealing_weights.shape == (self.len,)
        sample_probs *= annealing_weights
        assert sample_probs.shape == (self.len,)
        sample_probs = sample_probs / sum(sample_probs)
        assert sample_probs.shape == (self.len,)

        idxs = np.random.choice(self.len, size=batch_size, p=sample_probs, replace=False)
        return ReplayBatch(
            obs1=[self.obs1_buf[i] for i in idxs],
            acts=[self.acts_buf[i] for i in idxs],
            rews=[self.rews_buf[i] for i in idxs],
            obs2=[self.obs2_buf[i] for i in idxs],
            done=[self.done_buf[i] for i in idxs],
        )


class PrioritizedReplayBuffer(DumbPrioritizedReplayBuffer):
    def sample(self, batch_size=32) -> ReplayBatch:
        tds = self.tds_buf[:self.len]
        priorities = np.abs(tds) + self.eps

        priority_sum = np.sum(priorities)
        assert isinstance(priority_sum, np.float64)
        sample_probs = priorities / priority_sum
        assert sample_probs.shape == (self.len,)

        importsamp_ws = (1 / self.len * 1 / sample_probs) ** self.beta
        assert importsamp_ws.shape == (self.len,)
        importsamp_ws /= np.max(importsamp_ws)

        idxs = np.random.choice(self.len, size=batch_size, p=sample_probs, replace=False)

        return ReplayBatch(
            obs1=[self.obs1_buf[i] for i in idxs],
            acts=[self.acts_buf[i] for i in idxs],
            rews=[self.rews_buf[i] for i in idxs],
            obs2=[self.obs2_buf[i] for i in idxs],
            done=[self.done_buf[i] for i in idxs],
            impw=[importsamp_ws[i] for i in idxs],
            idxs=idxs
        )
