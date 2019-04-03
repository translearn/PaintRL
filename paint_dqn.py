from paint_ppo import main


if __name__ == '__main__':
    configuration = {
        'num_workers': 27,

        # 'model': {
        #     'use_lstm': False,
        # },
        'num_atoms': 1,
        'v_min': -100.0,
        'v_max': 100.0,
        'dueling': True,
        'double_q': True,
        'hiddens': [256, 128],

        'timesteps_per_iteration': 1000,
        'target_network_update_freq': 1000,
        'soft_q': False,
        'parameter_noise': False,
        'batch_mode': 'truncate_episodes',

        'buffer_size': 200000,
        'prioritized_replay': True,
        'compress_observations': False,

        'learning_starts': 1000,
        'sample_batch_size': 100,
        'train_batch_size': 512,

        'num_gpus': 1,
    }
    main('APEX', configuration)
