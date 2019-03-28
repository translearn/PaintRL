from paint_ppo import main


if __name__ == '__main__':
    configuration = {
        'num_workers': 27,

        'twin_q': True,
        'policy_delay': 2,
        'smooth_target_policy': True,

        # 'model': {
        #     # 'custom_model': 'paint_layer_model',
        #     # 'custom_options': {},  # extra options to pass to your model
        #     'use_lstm': False,
        # },

        'actor_hiddens': [256, 128],
        'critic_hiddens': [256, 128],

        'timesteps_per_iteration': 1000,
        'target_network_update_freq': 1000,
        'tau': 1e-3,

        'buffer_size': 200000,
        'prioritized_replay': True,

        'learning_starts': 1000,
        'sample_batch_size': 50,
        'train_batch_size': 512,

        'num_gpus': 1,
        # 'num_gpus_per_worker': 0,
    }
    main('APEX_DDPG', configuration)

