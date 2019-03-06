import os
import argparse
import ray
import ray.tune as tune
from ray.rllib.models import ModelCatalog
import ray.rllib.agents.a3c as a3c
from ray.rllib.rollout import rollout

from paint_ppo import call_backs, PaintModel, env_creator


ModelCatalog.register_custom_model('paint_model', PaintModel)
urdf_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv')
tune.registry.register_env('robot_gym_env', env_creator)


def make_a3c_env(is_train=True):
    conf = _make_configuration(is_train)
    return a3c.A3CAgent(env='robot_gym_env', config=conf)


def _make_configuration(is_train):
    env = {
        'urdf_root': urdf_root,
        'with_robot': False,
        'renders': False,
        'render_video': False,
        'rollout': False,
    }
    if not is_train:
        env['renders'] = True
        env['with_robot'] = False
        env['rollout'] = True

    conf = {
        'model': {
            'custom_model': 'paint_model',
            'custom_options': {},  # extra options to pass to your model
        },
        'num_workers': 5,

        'callbacks': call_backs,

        'env_config': env,
        'sample_batch_size': 10,

    }
    return conf


def train(config, reporter):
    _agent = a3c.A3CAgent(env='robot_gym_env', config=config)
    while True:
        result = _agent.train()
        reporter(**result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--path', type=str, default='/home/pyang/ray_results/')
    parser.add_argument('--warm-start', type=bool, default=False)
    args = parser.parse_args()
    ray.init(object_store_memory=5000000000, redis_max_memory=2000000000, log_to_driver=True)

    if args.mode == 'train':
        configuration = {
            'paint': {
                'run': train,
                'stop': {
                    'training_iteration': 10000,
                },
                'num_samples': 1,
                'config': _make_configuration(is_train=True),
                'checkpoint_freq': 200,

            }
        }
        trials = tune.run_experiments(configuration)

    else:
        agent = make_a3c_env(is_train=False)
        agent.restore(args.path)
        rollout(agent, 'robot_gym_env', 200)
