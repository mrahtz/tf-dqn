from functools import partial

from policies import mlp_features, cnn_features

default_config = dict(
    batch_size=32,
    gamma=1.0,
    random_action_prob=0.1,
    n_start_steps=1000,
    update_target_every_n_steps=500,
    log_every_n_steps=100,
    checkpoint_every_n_steps=1000,
    n_env_steps_per_rl_update=1,
    buffer_size=50000,
    lr=5e-4,
    render=False,
    env_id='CartPole-v0',
    train_n_steps=1e5,
    seed=0,
    double_dqn=True,
    dueling=True,
    prioritized=True,
    feature_extractor=partial(mlp_features, n_hidden=(64, 64)),
    gradient_clip=10
)

atari_config = dict(
    gamma=0.99,
    n_start_steps=10000,
    update_target_every_n_steps=1000,
    n_env_steps_per_rl_update=4,
    buffer_size=10000,
    lr=1e-4,
    feature_extractor=cnn_features,
    train_n_steps=1e7
)
