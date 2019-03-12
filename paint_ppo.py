import os
import argparse
import tensorflow as tf
import ray
import ray.tune as tune
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.rollout import run
from PaintRLEnv.robot_gym_env import RobotGymEnv


class PaintModel(Model):

    def _build_layers(self, inputs, num_outputs, options):
        pass

    def _build_layers_v2(self, input_dict, num_outputs, options):
        fc1 = tf.layers.dense(input_dict['obs'], 20, activation=tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 256, activation=tf.nn.relu, name='fc2')
        fc3 = tf.layers.dense(fc2, 128, activation=tf.nn.relu, name='fc3')
        out = tf.layers.dense(fc3, 4, activation=tf.nn.tanh, name='out')
        return out, fc3


def env_creator(env_config):
    return RobotGymEnv(**env_config)


ModelCatalog.register_custom_model('paint_model', PaintModel)
tune.registry.register_env('robot_gym_env', env_creator)


def on_episode_start(info):
    episode = info['episode']
    print('episode {} started'.format(episode.episode_id))
    episode.user_data['total_reward'] = 0
    episode.user_data['total_penalty'] = 0


def on_episode_step(info):
    episode = info['episode']
    episode_info = episode.last_info_for()
    if episode_info:
        episode.user_data['total_reward'] += episode_info['reward']
        episode.user_data['total_penalty'] += episode_info['penalty']


def on_episode_end(info):
    episode = info['episode']
    print('episode {} ended with length {}'.format(
        episode.episode_id, episode.length))
    episode.custom_metrics['total_reward'] = episode.user_data['total_reward']
    episode.custom_metrics['total_penalty'] = episode.user_data['total_penalty']
    episode.custom_metrics['total_return'] = episode.user_data['total_reward'] - episode.user_data['total_penalty']
    print('Achieved {0:.3f} return, in which {1:.3f} reward, '
          '{2:.3f} penalty in this episode.'.format(episode.custom_metrics['total_return'],
                                                    episode.custom_metrics['total_reward'],
                                                    episode.custom_metrics['total_penalty']))


def on_sample_end(info):
    print('returned sample batch of size {}'.format(info['samples'].count))


def on_train_result(info):
    print('agent.train() result: {} -> {} episodes'.format(
        info['agent'], info['result']['episodes_this_iter']))
    # you can mutate the result dict to add new fields to return
    info['result']['callback_ok'] = True


call_backs = {
            'on_episode_start': tune.function(on_episode_start),
            'on_episode_step': tune.function(on_episode_step),
            'on_episode_end': tune.function(on_episode_end),
            'on_sample_end': tune.function(on_sample_end),
            'on_train_result': tune.function(on_train_result),
}


def _make_env_config(is_train=True):
    env = {
        'urdf_root': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv'),
        'with_robot': False,
        'renders': False,
        'render_video': False,
        'rollout': False,
    }
    if not is_train:
        env['renders'] = True
        env['with_robot'] = False
        env['rollout'] = True
    return env


def main(algorithm, config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--checkpoint', type=str, default='/home/pyang/ray_results/paint')
    args = parser.parse_args()
    ray.init(object_store_memory=5000000000, redis_max_memory=2000000000, log_to_driver=False)
    experiment_config = {
        'paint': {
            'run': algorithm,
            'env': 'robot_gym_env',
            'stop': {
                'training_iteration': 10000,
            },
            'config': config,
            'checkpoint_freq': 100,

        }
    }
    experiment_config['paint']['config']['callbacks'] = call_backs
    if args.mode == 'train':
        experiment_config['paint']['config']['env_config'] = _make_env_config()
        tune.run_experiments(experiment_config, resume=False)
    else:
        args.run = experiment_config['paint']['run']
        args.env = experiment_config['paint']['env']
        args.steps = 200
        experiment_config['paint']['config']['env_config'] = _make_env_config(is_train=False)
        args.config = experiment_config['paint']['config']
        args.out = None
        args.no_render = True
        run(args, parser)


if __name__ == '__main__':
    configuration = {
        'num_workers': 5,

        'simple_optimizer': False,

        'model': {
            'custom_model': 'paint_model',
            'custom_options': {},  # extra options to pass to your model
        },
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',
        'vf_share_layers': True,

        'sample_batch_size': 200,
        'train_batch_size': 1600,
        'sgd_minibatch_size': 64,
        'num_sgd_iter': 30,
    }
    main('PPO', configuration)
