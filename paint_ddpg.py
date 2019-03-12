from paint_ppo import main


if __name__ == '__main__':
    configuration = {
        'num_workers': 5,

        'twin_q': True,
        'policy_delay': 1,
        'smooth_target_policy': True,

        'actor_hiddens': [256, 128],
        'critic_hiddens': [256, 128],

        'timesteps_per_iteration': 1000,
        'target_network_update_freq': 1000,
        'tau': 1e-3,

        'buffer_size': 50000,
        'prioritized_replay': True,

        'learning_starts': 2000,
        'sample_batch_size': 50,
        'train_batch_size': 512,

        'num_gpus': 1,
        'num_gpus_per_worker': 0,
    }
    main('APEX_DDPG', configuration)

