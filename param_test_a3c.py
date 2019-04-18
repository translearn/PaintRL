from param_test_ppo import main


if __name__ == '__main__':
    configuration = {
        'model': {
            'fcnet_hiddens': [512],
            'use_lstm': False,
        },
        'num_workers': 11,
        'sample_batch_size': 10,

        'vf_loss_coeff': 1.0,
        'entropy_coeff': 0.0,
    }
    main('A3C', configuration)
