import argparse
import ray
import ray.tune as tune
from ray.rllib.rollout import run
from PaintRLEnv.param_test_env import ParamTestEnv


def main(algorithm, config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--checkpoint', type=str, default='/home/pyang/ray_results/paint')
    args = parser.parse_args()

    def env_creator(env_config):
        return ParamTestEnv(**env_config)

    tune.registry.register_env('param_test_env', env_creator)

    experiment_config = {
        'param_test': {
            'run': algorithm,
            'env': 'param_test_env',
            'stop': {
                'training_iteration': 10000,
            },
            'config': config,
            'checkpoint_freq': 500,
        }
    }
    experiment_config['param_test']['config']['env_config'] = {'size': 22, }
    if args.mode == 'train':
        ray.init(object_store_memory=10000000000, redis_max_memory=10000000000, log_to_driver=True)
        # ray.init(redis_address='141.3.81.141:6379')
        tune.run_experiments(experiment_config)
    else:
        experiment_config['param_test']['config']['num_workers'] = 2
        args.run = experiment_config['param_test']['run']
        args.env = experiment_config['param_test']['env']
        args.steps = 400
        args.config = experiment_config['param_test']['config']
        args.out = None
        args.no_render = True
        run(args, parser)


if __name__ == '__main__':
    configuration = {
        'num_workers': 15,
        'num_gpus': 1,
        'simple_optimizer': False,

        'model': {
            'fcnet_hiddens': [256, 128],
            'use_lstm': False,
        },
        'vf_share_layers': False,
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',
        'vf_clip_param': 2,

        'sample_batch_size': 100,
        'train_batch_size': 1500,
        'sgd_minibatch_size': 64,
        'num_sgd_iter': 16,
    }
    main('PPO', configuration)
