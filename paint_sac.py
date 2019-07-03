from paint_ppo import main

if __name__ == '__main__':
    configuration = {
        'num_workers': 15,
        'num_gpus': 1,

        'Q_model': {
            'fcnet_hiddens': (256, 128),
        },
        'policy_model': {

            'fcnet_hiddens': (256, 128),
        },

        'n_step': 1,
        
        'evaluation_interval': None,
        'evaluation_num_episodes': 1,

        'timesteps_per_iteration': 1000,
        'tau': 5e-3,

        'buffer_size': 200000,
        'prioritized_replay': False,

        'optimization': {
            'learning_rate': 5e-4,

            'policy_loss_weight': 1.0,
            'Q_loss_weight': 1.0,
            'entropy_loss_weight': 1.0,
        },
        'learning_starts': 1000,

        'sample_batch_size': 20,
        'train_batch_size': 300,

        'worker_side_prioritization': False,
        'min_iter_time_s': 1,

    }
    main('SAC', configuration)
