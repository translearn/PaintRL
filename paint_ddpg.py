import os
import argparse
import ray
import ray.tune as tune
from ray.rllib.agents.ddpg import ApexDDPGAgent
from ray.rllib.rollout import rollout
from paint_ppo import call_backs, env_creator


urdf_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv')
tune.registry.register_env('robot_gym_env', env_creator)


def make_ddpg_env(is_train=True):
    conf = _make_configuration(is_train)
    return ApexDDPGAgent(env='robot_gym_env', config=conf)


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
        'num_workers': 10,

        'callbacks': call_backs,

        'env_config': env,

        'twin_q': True,
        'policy_delay': 1,
        'smooth_target_policy': True,

        'actor_hiddens': [256, 128],
        'critic_hiddens': [256, 128],

        'timesteps_per_iteration': 1000,
        'target_network_update_freq': 1000,
        'tau': 1e-3,

        'buffer_size': 50000,
        'prioritized_replay': True,

        'learning_starts': 2000,
        'sample_batch_size': 50,
        'train_batch_size': 512,

        'num_gpus': 1,
        'num_gpus_per_worker': 0,
    }
    return conf


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
                'run': 'APEX_DDPG',
                'env': 'robot_gym_env',
                'stop': {
                    'training_iteration': 10000,
                },
                # 'resources_per_trial': {
                #     'cpu': 0,
                #     'gpu': 0,
                # },
                'num_samples': 1,
                'config': _make_configuration(is_train=True),
                'checkpoint_freq': 100,

            }
        }
        tune.run_experiments(configuration, resume=False)

    else:
        agent = make_ddpg_env(is_train=False)
        agent.restore(args.path)
        rollout(agent, 'robot_gym_env', 200)
