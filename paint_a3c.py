from paint_ppo import main


if __name__ == '__main__':
    configuration = {
        # 'model': {
        #     'custom_model': 'paint_model',
        #     'custom_options': {},
        # },
        'model': {
            'fcnet_hiddens': [256, 128],
        },
        'num_workers': 10,
        'sample_batch_size': 10,
    }
    main('A3C', configuration)

