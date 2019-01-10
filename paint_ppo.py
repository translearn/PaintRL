import os
import argparse
import tensorflow as tf
import ray
import ray.tune as tune
# from ray.tune.logger import pretty_print
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.rollout import rollout
from PaintRLEnv.robot_gym_env import RobotGymEnv


class PaintModel(Model):

    def _build_layers(self, inputs, num_outputs, options):
        pass

    def _build_layers_v2(self, input_dict, num_outputs, options):
        fc1 = tf.layers.dense(input_dict['obs'], 20, activation=tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name='fc2')
        fc3 = tf.layers.dense(fc2, 128, activation=tf.nn.relu, name='fc3')
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
    # episode.user_data['robot_pose'] = []


def on_episode_step(info):
    episode = info['episode']
    robot_pose = episode.last_observation_for()[-2:]
    # episode.user_data['robot_pose'].append(robot_pose)


def on_episode_end(info):
    episode = info['episode']
    # pole_angle = np.mean(episode.user_data['robot_pose'])
    print('episode {} ended with length {}'.format(
        episode.episode_id, episode.length))
    # episode.custom_metrics['robot_pose'] = pole_angle


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
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--path', type=str, default='/home/pyang/ray_results/')
    args = parser.parse_args()
    ray.init()

    agent = ppo.PPOAgent(env='robot_gym_env', config={
        'num_workers': 4,
        'simple_optimizer': False,
        'callbacks': {
            'on_episode_start': tune.function(on_episode_start),
            'on_episode_step': tune.function(on_episode_step),
            'on_episode_end': tune.function(on_episode_end),
            'on_sample_end': tune.function(on_sample_end),
            'on_train_result': tune.function(on_train_result),
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
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',
        'vf_share_layers': True,
        'num_gpus': 1,
        'num_gpus_per_worker': 0.25,
        # 'lr_schedule': [[0, 1e-3],
        #                 [1e7, 1e-12], ],
        'sample_batch_size': 200,
        'train_batch_size': 800,
        'sgd_minibatch_size': 32,
        'num_sgd_iter': 30,
    })
    # conf = ppo.DEFAULT_CONFIG.copy()
    configuration = {
        'paint': {
            'run': train,
            # 'stop': {
            #     'training_iteration': args.num_iters,
            # },
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
                    'on_train_result': tune.function(on_train_result),
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
                'num_workers': 0,
                'simple_optimizer': True,
                'observation_filter': 'NoFilter',
                'vf_share_layers': True,
                'num_gpus': 1,
                'num_gpus_per_worker': 1,
                'sample_batch_size': 100,
                'train_batch_size': 200,
                'sgd_minibatch_size': 5,
                'num_sgd_iter': 10,
            },
        }
    }
    if args.mode == 'train':
        counter = 1
        max_reward = 0
        while True:
            counter += 1
            res = agent.train()
            # print(pretty_print(res))
            if counter % 500 == 0:
                model_path = agent.save()
                print('model saved at:{} in step {}'.format(model_path, counter))
            if max_reward >= 5000 and res['episode_reward_mean'] >= 2000:
                model_path = agent.save()
                print('max rewards already reached 50%, stop training, model saved at:{}'.format(model_path))
                break
            else:
                if res['episode_reward_max'] > max_reward:
                    max_reward = res['episode_reward_max']
                print('maximum reward currently:{}'.format(max_reward))
    else:
        agent.restore(args.path)
        # try to use the model
        # try the rollout function
        rollout(agent, 'robot_gym_env', 200)

    # trials = tune.run_experiments(configuration)

    # # verify custom metrics for integration tests
    # custom_metrics = trials[0].last_result['custom_metrics']
    # print(custom_metrics)
    # assert 'pole_angle_mean' in custom_metrics
    # assert 'pole_angle_min' in custom_metrics
    # assert 'pole_angle_max' in custom_metrics
    # assert type(custom_metrics['pole_angle_mean']) is float
    # assert 'callback_ok' in trials[0].last_result
