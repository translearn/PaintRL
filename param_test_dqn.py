from param_test_ppo import main


if __name__ == '__main__':
    configuration = {
        # 'num_workers': 6,
        'num_gpus': 1,

        'double_q': False,
        'dueling': False,
        'num_atoms': 1,
        'noisy': False,
        'prioritized_replay': False,
        'n_step': 1,
        'target_network_update_freq': 8000,
        'lr': .0000625,
        'adam_epsilon': .00015,
        'hiddens': [512],
        'learning_starts': 20000,
        'buffer_size': 1000000,
        'sample_batch_size': 4,
        'train_batch_size': 32,
        'schedule_max_timesteps': 2000000,
        'exploration_final_eps': 0.01,
        'exploration_fraction': 0.1,
        'prioritized_replay_alpha': 0.5,
        'beta_annealing_fraction': 1.0,
        'final_prioritized_replay_beta': 1.0,
        'timesteps_per_iteration': 10000,
    }
    main('DQN', configuration)
