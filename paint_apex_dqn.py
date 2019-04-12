from paint_ppo import main


if __name__ == '__main__':
    configuration = {
        'num_workers': 27,
        'num_gpus': 1,

        'num_atoms': 1,
        # 'v_min': -120.0,
        # 'v_max': 120.0,
        'dueling': True,
        'double_q': True,
        'hiddens': [256],

        'timesteps_per_iteration': 200,
        'target_network_update_freq': 1000,
        'soft_q': False,
        'parameter_noise': False,
        'batch_mode': 'truncate_episodes',

        'buffer_size': 10000,
        'prioritized_replay': False,
        'compress_observations': False,

        'learning_starts': 1000,
        'sample_batch_size': 5,
        'train_batch_size': 24,
    }
    main('APEX', configuration)