import os
import argparse
import tensorflow as tf
import numpy as np
import ray
import ray.tune as tune
from ray.tune.logger import pretty_print
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import flatten
from PaintRLEnv.robot_gym_env import RobotGymEnv


def _get_pre_trained_graph_output():
    resnet_path = './resnet_v2_fp32_savedmodel_NCHW/1538687196'
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], resnet_path)
    return sess.graph_def


pretrained_graph = _get_pre_trained_graph_output()


class PaintModel(Model):

    def _build_layers(self, inputs, num_outputs, options):
        pass

    def _build_layers_v2(self, input_dict, num_outputs, options):
        scaled_images = tf.cast(input_dict['obs']['image'], tf.float32) / 224.
        output_tensor = tf.import_graph_def(pretrained_graph, input_map={'input_tensor': scaled_images},
                                            return_elements=['resnet_model/Relu_48:0'])[0]
        # output_tensor = resnet_graph.get_tensor_by_name('resnet_model/Relu_48:0')
        print()
        output_tensor = flatten(output_tensor)
        # fc1 = tf.layers.dense(conv3, 512, activation=tf.nn.relu, name='fc1')
        fc1 = tf.concat([output_tensor, input_dict['obs']['pose']], 1)
        fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name='fc2')
        fc3 = tf.layers.dense(fc2, 32, activation=tf.nn.relu, name='fc3')
        out = tf.layers.dense(fc3, 4, activation=tf.nn.tanh, name='out')
        return out, fc3


def env_creator(env_config):
    return RobotGymEnv(**env_config)


urdf_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv')

ModelCatalog.register_custom_model('paint_model', PaintModel)
tune.registry.register_env('robot_gym_env', env_creator)


def on_episode_start(info):
    episode = info['episode']
    print('episode {} started'.format(episode.episode_id))
    episode.user_data['pole_angles'] = []


def on_episode_step(info):
    episode = info['episode']
    pole_angle = abs(episode.last_observation_for()[2])
    episode.user_data['pole_angles'].append(pole_angle)


def on_episode_end(info):
    episode = info['episode']
    pole_angle = np.mean(episode.user_data['pole_angles'])
    print('episode {} ended with length {} and pole angles {}'.format(
        episode.episode_id, episode.length, pole_angle))
    episode.custom_metrics['pole_angle'] = pole_angle


def on_sample_end(info):
    print('returned sample batch of size {}'.format(info['samples'].count))


def on_train_result(info):
    print('agent.train() result: {} -> {} episodes'.format(
        info['agent'], info['result']['episodes_this_iter']))
    # you can mutate the result dict to add new fields to return
    info['result']['callback_ok'] = True


def train(config, reporter):
    agent = ppo.PPOAgent(config=config, env='robot_gym_env')
    while True:
        result = agent.train()
        reporter(**result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-iters', type=int, default=2000)
    args = parser.parse_args()
    ray.init()

    agent = ppo.PPOAgent(env='robot_gym_env', config={
        'num_workers': 1,
        'callbacks': {
            'on_episode_start': tune.function(on_episode_start),
            'on_episode_step': tune.function(on_episode_step),
            'on_episode_end': tune.function(on_episode_end),
            'on_sample_end': tune.function(on_sample_end),
            # wait for version 0.7 release
            # 'on_train_result': tune.function(on_train_result),
        },
        'model': {
            'custom_model': 'paint_model',
            'custom_options': {},  # extra options to pass to your model
        },
        'env_config': {
            'urdf_root': urdf_root,
            'renders': False,
            'render_video': False,
        },
        'observation_filter': 'NoFilter',
        'vf_share_layers': True,
        'num_gpus': 1,
        'num_gpus_per_worker': 1,
        'sample_batch_size': 50,
        'train_batch_size': 50,
        'sgd_minibatch_size': 32,
        # what exactly is this?
        'num_sgd_iter': 2,
    })
    # conf = ppo.DEFAULT_CONFIG.copy()
    configuration = {
        'paint': {
            'run': train,
            'stop': {
                'training_iteration': args.num_iters,
            },
            'trial_resources': {
                'cpu': 1,
                'gpu': 1,
            },
            'num_samples': 1,
            'config': {
                'callbacks': {
                    'on_episode_start': tune.function(on_episode_start),
                    'on_episode_step': tune.function(on_episode_step),
                    'on_episode_end': tune.function(on_episode_end),
                    'on_sample_end': tune.function(on_sample_end),
                    # wait for version 0.7 release
                    # 'on_train_result': tune.function(on_train_result),
                },
                'model': {
                    'custom_model': 'paint_model',
                    'custom_options': {},  # extra options to pass to your model
                },
                'env_config': {
                    'urdf_root': urdf_root,
                    'renders': False,
                    'render_video': False,
                },
                'observation_filter': 'NoFilter',
                'vf_share_layers': True,
                'num_gpus': 1,
                'num_gpus_per_worker': 1,
                'sample_batch_size': 50,
                'train_batch_size': 50,
                'sgd_minibatch_size': 32,
                # what exactly is this?
                'num_sgd_iter': 2,
            },
        }
    }

    for _ in range(100):
        res = agent.train()
        print(pretty_print(res))

    # trials = tune.run_experiments(configuration)

    # # verify custom metrics for integration tests
    # custom_metrics = trials[0].last_result['custom_metrics']
    # print(custom_metrics)
    # assert 'pole_angle_mean' in custom_metrics
    # assert 'pole_angle_min' in custom_metrics
    # assert 'pole_angle_max' in custom_metrics
    # assert type(custom_metrics['pole_angle_mean']) is float
    # assert 'callback_ok' in trials[0].last_result
