from param_test_ppo import main


if __name__ == '__main__':
    configuration = {
        'num_workers': 31,
        'num_gpus': 1,

        'num_atoms': 1,
        'dueling': False,
        'double_q': False,
        'hiddens': [256, 128],

        'n_step': 3,
        'lr': 0.0001,
        'adam_epsilon': .00015,

        'timesteps_per_iteration': 5000,
        'target_network_update_freq': 10000,
        'soft_q': False,
        'parameter_noise': False,
        'batch_mode': 'truncate_episodes',

        'schedule_max_timesteps': 2000000,
        'exploration_final_eps': 0.01,
        'exploration_fraction': .1,
        'prioritized_replay_alpha': 0.5,
        'beta_annealing_fraction': 1.0,
        'final_prioritized_replay_beta': 1.0,
        'buffer_size': 200000,
        'prioritized_replay': True,

        # 'learning_starts': 1000,
        'sample_batch_size': 20,
        'train_batch_size': 512,

    }
    main('APEX', configuration)
