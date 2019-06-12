params = dict(
    batch_size=32,
    gamma=1.0,
    random_action_prob=0.1,
    n_start_steps=1000,
    update_target_every_n_steps=500,
    log_every_n_steps=100,
    checkpoint_every_n_steps=1000,
    buffer_size=50000,
    lr=5e-4,
    render=False,
    env_id='CartPole-v0',
    train_n_steps=100000,
    seed=0,
    double_dqn=False,
    dueling_dqn=False
)
