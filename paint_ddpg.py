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


# def train(config, reporter):
#     agent = ppo.PPOAgent(config=config, env='robot_gym_env')
#     while True:
#         result = agent.train()
#         reporter(**result)


def make_ddpg_env(is_train=True, with_lr_schedule=False):
    workers = 10
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
        workers = 2

    # lr_schedule = None
    if with_lr_schedule:
        print()

    ddpg_agent = ddpg.ApexDDPGAgent(env='robot_gym_env', config={
        'num_workers': workers,

        'callbacks': {
            'on_episode_start': tune.function(on_episode_start),
            'on_episode_step': tune.function(on_episode_step),
            'on_episode_end': tune.function(on_episode_end),
            'on_sample_end': tune.function(on_sample_end),
            'on_train_result': tune.function(on_train_result),
        },

        'env_config': env,

        'twin_q': True,
        'policy_delay': 2,
        'smooth_target_policy': True,

        'actor_hiddens': [256, 128],
        'critic_hiddens': [256, 128],

        'timesteps_per_iteration': 1000,
        'target_network_update_freq': 10000,
        'tau': 1e-3,

        'buffer_size': 50000,
        'prioritized_replay': True,

        # 'use_huber': True,
        # 'huber_threshold': 1.0,
        'learning_starts': 10000,
        'sample_batch_size': 50,
        'train_batch_size': 128,

        'num_gpus': num_gpus,
        'num_gpus_per_worker': num_gpus / workers if workers != 0 else 0,

        'compress_observations': True,
    })

    return ddpg_agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--path', type=str, default='/home/pyang/ray_results/')
    parser.add_argument('--warm-start', type=bool, default=False)
    args = parser.parse_args()
    ray.init()

    if args.mode == 'train':
        # counter = 1
        agent = make_ddpg_env()
        if args.warm_start:
            agent.restore(args.path)
            print('warm started from path {}'.format(args.path))
        for i in range(10000):
            # counter += 1
            res = agent.train()
            # print(pretty_print(res))
            if i % 200 == 0:
                model_path = agent.save()
                print('model saved at:{} in step {}'.format(model_path, i))
            # if res['episode_reward_max'] >= 9000 and res['episode_reward_mean'] >= 7500:
            #     model_path = agent.save()
            #     print('max rewards already reached 50%, stop training, model saved at:{}'.format(model_path))
            #     break
            else:
                print('current training step:{}'.format(i))
                print('maximum reward currently:{0:.3f}'.format(res['episode_reward_max']))
    else:
        agent = make_ddpg_env(is_train=False)
        agent.restore(args.path)
        rollout(agent, 'robot_gym_env', 200)
