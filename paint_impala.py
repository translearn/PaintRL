from paint_ppo import main


if __name__ == '__main__':

    configuration = {
        'model': {
            'fcnet_hiddens': [256, 128],
            'use_lstm': False,
        },
        'num_workers': 15,
        'num_gpus': 1,

        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',

        'lr': 0.0005,
        'sample_batch_size': 50,
        'train_batch_size': 750,

        # 'num_sgd_iter': 16,
        "num_data_loader_buffers": 4,
        # how many train batches should be retained for minibatching. This conf
        # only has an effect if `num_sgd_iter > 1`.
        "minibatch_buffer_size": 4,

        # set >0 to enable experience replay. Saved samples will be replayed with
        # a p:1 proportion to new data samples.
        # 'replay_proportion': 10,
        # number of sample batches to store for replay. The number of transitions
        # saved total will be (replay_buffer_num_slots * sample_batch_size).
        # 'replay_buffer_num_slots': 100,

        # level of queuing for sampling.
        'max_sample_requests_in_flight_per_worker': 1,

        'broadcast_interval': 3,

        'grad_clip': 40.0,

        'vf_loss_coeff': 0.5,
        'entropy_coeff': 0.01,
    }
    main('IMPALA', configuration)
