from paint_ppo import main


if __name__ == '__main__':

    configuration = {
        'model': {
            'custom_model': 'paint_model',
            'custom_options': {},
        },
        'num_workers': 5,
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',

        'lr': 0.0005,
        'sample_batch_size': 50,
        'train_batch_size': 800,
        'min_iter_time_s': 10,

        'num_sgd_iter': 30,
        "num_data_loader_buffers": 2,
        # how many train batches should be retained for minibatching. This conf
        # only has an effect if `num_sgd_iter > 1`.
        "minibatch_buffer_size": 1,

        # set >0 to enable experience replay. Saved samples will be replayed with
        # a p:1 proportion to new data samples.
        'replay_proportion': 10,
        # number of sample batches to store for replay. The number of transitions
        # saved total will be (replay_buffer_num_slots * sample_batch_size).
        'replay_buffer_num_slots': 100,

        # level of queuing for sampling.
        'max_sample_requests_in_flight_per_worker': 2,

        'broadcast_interval': 1,

        'grad_clip': 40.0,

        'opt_type': 'adam',

        'vf_loss_coeff': 0.5,
        'entropy_coeff': -0.01,
    }
    main('IMPALA', configuration)
