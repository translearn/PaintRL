from paint_ppo import main


if __name__ == '__main__':
    configuration = {
        'num_workers': 6,

        # 'model': {
        #     'use_lstm': False,
        # },
        'num_atoms': 1,
        'v_min': -120.0,
        'v_max': 120.0,
        'dueling': True,
        'double_q': True,
        'hiddens': [256, 128],

        'timesteps_per_iteration': 100,
        'target_network_update_freq': 1000,
        'soft_q': False,
        'parameter_noise': False,
        'batch_mode': 'truncate_episodes',

        'buffer_size': 10000,
        'prioritized_replay': False,
        'compress_observations': False,

        'learning_starts': 100,
        'sample_batch_size': 1,
        'train_batch_size': 24,

        'num_gpus': 1,
    }
    main('APEX', configuration)
