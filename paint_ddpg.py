import os
import argparse
import ray
import ray.tune as tune
import ray.rllib.agents.ddpg as ddpg
from ray.rllib.rollout import rollout
from PaintRLEnv.robot_gym_env import RobotGymEnv


def env_creator(env_config):
    return RobotGymEnv(**env_config)


urdf_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv')
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


def make_ddpg_env(is_train=True):
    conf = _make_configuration(is_train)
    return ddpg.ApexDDPGAgent(env='robot_gym_env', config=conf)


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
        'num_workers': 5,

        'callbacks': {
            'on_episode_start': tune.function(on_episode_start),
            'on_episode_step': tune.function(on_episode_step),
            'on_episode_end': tune.function(on_episode_end),
            'on_sample_end': tune.function(on_sample_end),
            'on_train_result': tune.function(on_train_result),
        },

        'env_config': env,

        'twin_q': True,
        'policy_delay': 1,
        'smooth_target_policy': True,

        'actor_hiddens': [256, 128],
        'critic_hiddens': [256, 128],

        'timesteps_per_iteration': 1000,
        'target_network_update_freq': 1000,
        'tau': 1e-3,

        'buffer_size': 20000,
        'prioritized_replay': True,

        'learning_starts': 2000,
        'sample_batch_size': 50,
        'train_batch_size': 512,

        'num_gpus': 1,
        'num_gpus_per_worker': 0,

        # 'compress_observations': True,
    }
    return conf


def train(config, reporter):
    _agent = ddpg.ApexDDPGAgent(env='robot_gym_env', config=config)
    while True:
        result = _agent.train()
        reporter(**result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--path', type=str, default='/home/pyang/ray_results/')
    parser.add_argument('--warm-start', type=bool, default=False)
    args = parser.parse_args()
    ray.init(object_store_memory=5000000000, redis_max_memory=2000000000, log_to_driver=False)

    if args.mode == 'train':
        configuration = {
            'paint': {
                'run': train,
                'stop': {
                    'training_iteration': 10000,
                },
                # 'resources_per_trial': {
                #     'cpu': 0,
                #     'gpu': 0,
                # },
                'num_samples': 1,
                'config': _make_configuration(is_train=True),
                'checkpoint_freq': 200,

            }
        }
        trials = tune.run_experiments(configuration)

    else:
        agent = make_ddpg_env(is_train=False)
        agent.restore(args.path)
        rollout(agent, 'robot_gym_env', 200)
    #
    # if args.mode == 'train':
    #     agent = make_ddpg_env()
    #     if args.warm_start:
    #         agent.restore(args.path)
    #         print('warm started from path {}'.format(args.path))
    #     for i in range(10000):
    #         res = agent.train()
    #         if i % 200 == 0:
    #             model_path = agent.save()
    #             print('model saved at:{} in step {}'.format(model_path, i))
    #         else:
    #             print('current training step:{}'.format(i))
    #             print('maximum reward currently:{0:.3f}'.format(res['episode_reward_max']))
    # else:
    #     agent = make_ddpg_env(is_train=False)
    #     agent.restore(args.path)
    #     rollout(agent, 'robot_gym_env', 200)
