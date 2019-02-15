import ray
import ray.tune as tune
from ray.rllib.agents.ddpg import ApexDDPGAgent
import gym


class TestEnv(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make('MountainCarContinuous-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


def env_creator(env_config):
    return TestEnv(env_config)


tune.registry.register_env("testenv", env_creator)


if __name__ == '__main__':

    ray.init()

    agent = ApexDDPGAgent(env='testenv', config={
        'num_workers': 4,

        'env_config': {},

        'twin_q': True,
        'policy_delay': 2,
        'smooth_target_policy': True,

        'actor_hiddens': [256, 128],
        'critic_hiddens': [256, 128],

        'timesteps_per_iteration': 600,
        'target_network_update_freq': 10000,
        'tau': 1e-3,

        'buffer_size': 20000,
        'prioritized_replay': True,

        'learning_starts': 10000,
        'sample_batch_size': 50,
        'train_batch_size': 64,

        'num_gpus': 1,
        'num_gpus_per_worker': 1 / 4,

        'compress_observations': True,
    })

    for i in range(10000):
        agent.train()

