import os
import argparse
import tensorflow as tf
import ray
import ray.tune as tune
import ray.rllib.agents.impala as impala
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.rollout import rollout
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


urdf_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv')

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


# def train(config, reporter):
#     agent = ppo.PPOAgent(config=config, env='robot_gym_env')
#     while True:
#         result = agent.train()
#         reporter(**result)


def make_impala_env(is_train=True, with_lr_schedule=False):
    workers = 4
    num_gpus = 1
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
        workers = 0

    lr_schedule = None
    if with_lr_schedule:
        lr_schedule = [[0, 1e-3], [1e5, 1e-7], ]

    impala_agent = impala.ImpalaAgent(env='robot_gym_env', config={
        'num_workers': workers,
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
        'env_config': env,
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',
        'num_gpus': num_gpus,
        'num_gpus_per_worker': 0,
        'lr': 0.0005,
        'lr_schedule': lr_schedule,
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
    })
    return impala_agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--path', type=str, default='/home/pyang/ray_results/')
    parser.add_argument('--warm-start', type=bool, default=False)
    args = parser.parse_args()
    ray.init()

    if args.mode == 'train':
        agent = make_impala_env()
        if args.warm_start:
            agent.restore(args.path)
            print('warm started from path {}'.format(args.path))
        for i in range(10000):
            res = agent.train()
            if i % 200 == 0:
                model_path = agent.save()
                print('model saved at:{} in step {}'.format(model_path, i))
            else:
                print('current training step:{}'.format(i))
                print('maximum reward currently:{0:.3f}'.format(res['episode_reward_max']))
    else:
        agent = make_impala_env(is_train=False)
        agent.restore(args.path)
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

    # conf = ppo.DEFAULT_CONFIG.copy()
    # configuration = {
    #     'paint': {
    #         'run': train,
    #         # 'stop': {
    #         #     'training_iteration': args.num_iters,
    #         # },
    #         'trial_resources': {
    #             'cpu': 1,
    #             'gpu': 1,
    #         },
    #         'num_samples': 1,
    #         'config': {
    #             'callbacks': {
    #                 'on_episode_start': tune.function(on_episode_start),
    #                 'on_episode_step': tune.function(on_episode_step),
    #                 'on_episode_end': tune.function(on_episode_end),
    #                 'on_sample_end': tune.function(on_sample_end),
    #                 'on_train_result': tune.function(on_train_result),
    #             },
    #             'model': {
    #                 'custom_model': 'paint_model',
    #                 'custom_options': {},  # extra options to pass to your model
    #             },
    #             'env_config': {
    #                 'urdf_root': urdf_root,
    #                 'renders': False,
    #                 'render_video': False,
    #             },
    #             'num_workers': 0,
    #             'simple_optimizer': True,
    #             'observation_filter': 'NoFilter',
    #             'vf_share_layers': True,
    #             'num_gpus': 1,
    #             'num_gpus_per_worker': 1,
    #             'sample_batch_size': 100,
    #             'train_batch_size': 200,
    #             'sgd_minibatch_size': 5,
    #             'num_sgd_iter': 10,
    #         },
    #     }
    # }
