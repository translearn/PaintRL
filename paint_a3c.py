from paint_ppo import main


if __name__ == '__main__':
    configuration = {
        # 'model': {
        #     'custom_model': 'paint_model',
        #     'custom_options': {},
        # },
        'model': {
            'fcnet_hiddens': [256, 128],
            'use_lstm': False,
        },
        'num_workers': 15,
        'sample_batch_size': 20,
    }
    main('A3C', configuration)

