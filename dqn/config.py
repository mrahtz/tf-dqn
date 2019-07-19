# https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/deepq/dqn.py
default_config = dict(
    batch_size=32,
    gamma=0.99,
    random_action_prob=0.1,
    n_start_steps=1_000,
    update_target_every_n_steps=500,
    log_every_n_steps=100,
    checkpoint_every_n_steps=1_000,
    n_env_steps_per_rl_update=1,
    buffer_size=50_000,
    lr=5e-4,
    render=False,
    env_id='CartPole-v0',
    train_n_steps=1e5,
    seed=0,
    double_dqn=True,
    dueling=True,
    prioritized=False,
    features='mlp',
    gradient_clip=10,
    async_test=False,
    test_every_n_steps=5_000,
)

# https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/deepq/experiments/run_atari.py
atari_config = dict(
    lr=1e-4,
    buffer_size=10_000,
    update_target_every_n_steps=1_000,
    n_start_steps=10_000,
    n_env_steps_per_rl_update=4,
)
