from paint_ppo import main


if __name__ == '__main__':
    configuration = {
        'num_workers': 15,
        'num_gpus': 1,

        'num_atoms': 1,
        # 'v_min': -120.0,
        # 'v_max': 120.0,
        'dueling': True,
        'double_q': True,
        'hiddens': [256, 128],

        'exploration_final_eps': 0.01,
        "schedule_max_timesteps": 2000000,
        'exploration_fraction': 0.2,

        'timesteps_per_iteration': 1000,
        'target_network_update_freq': 3000,
        'soft_q': False,
        'parameter_noise': False,
        'batch_mode': 'truncate_episodes',

        'buffer_size': 200000,
        'prioritized_replay': True,
        'compress_observations': False,

        'learning_starts': 1000,
        'sample_batch_size': 20,
        'train_batch_size': 32,
    }
    main('APEX', configuration)
