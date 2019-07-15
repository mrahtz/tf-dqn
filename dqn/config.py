default_config = dict(
    batch_size=32,
    gamma=1.0,
    random_action_prob=0.1,
    n_start_steps=1000,
    update_target_every_n_steps=500,
    log_every_n_steps=100,
    checkpoint_every_n_steps=1000,
    n_env_steps_per_rl_update=1,
    buffer_size=50_000,
    lr=5e-4,
    render=False,
    env_id='CartPole-v0',
    train_n_steps=1e5,
    seed=0,
    double_dqn=True,
    dueling=True,
    prioritized=True,
    features='mlp',
    gradient_clip=10
)

# Hyperparams from
# https://github.com/hill-a/stable-baselines/blob/72dab6a37bb33ad959cc598f187299edc49c425f/stable_baselines/deepq/dqn.py
atari_config = dict(
    gamma=0.99,
    n_start_steps=1_000,
    update_target_every_n_steps=500,
    n_env_steps_per_rl_update=1,
    feature_extractor='cnn',
    train_n_steps=1e7
)
