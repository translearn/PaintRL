import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class ParamTestEnv(gym.Env):

    action_space = spaces.Discrete(4)
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float64)

    EPISODE_MAX_LENGTH = 500

    def __init__(self, size):
        self._size = size
        self._world = {}
        self._i = 1
        self._j = 1
        self._reward_counter = 0
        self._step_counter = 0
        self._violated_wall = False
        for i in range(self._size):
            for j in range(self._size):
                if i in (0, self._size - 1) or j in (0, self._size - 1):
                    self._world[(i, j)] = 0
                else:
                    self._world[(i, j)] = 1
                    self._reward_counter += 1
        self._init_world = self._world.copy()
        self._init_reward_counter = self._reward_counter

    def reset(self):
        self._i = 1
        self._j = 1
        self._violated_wall = False
        self._reward_counter = self._init_reward_counter
        self._step_counter = 0
        self._world = self._init_world.copy()
        return self._observation()

    def _step(self, action):
        immediate_reward = self._get_immediate_reward()
        self._step_counter += 1
        if action == 0:
            self._i += 1
        elif action == 1:
            self._j += 1
        elif action == 2:
            self._i -= 1
        elif action == 3:
            self._j -= 1
        else:
            raise IndexError('No such action!')
        if self._i < 0 or self._i >= self._size or self._j < 0 or self._j >= self._size:
            self._i = self._clip_pos(self._i)
            self._j = self._clip_pos(self._j)
            self._violated_wall = True
        return immediate_reward

    def _clip_pos(self, pos):
        if pos < 0:
            pos = 0
        if pos >= self._size:
            pos = self._size - 1
        return pos

    def _termination(self):
        if self._violated_wall or self._reward_counter <= 0 or self._step_counter >= self.EPISODE_MAX_LENGTH - 1:
            return True
        return False

    def _observation(self):

        i = self._i / self._size
        j = self._j / self._size
        return np.asarray([i, j])

    def _get_immediate_reward(self):
        if self._world[(self._i, self._j)] > 0:
            self._world[(self._i, self._j)] -= 1
            self._reward_counter -= 1
            return 1
        return 0

    def _reward(self):
        if self._violated_wall:
            return 0
        return self._get_immediate_reward()

    def step(self, action):
        immediate_reward = self._step(action)
        reward = self._reward()
        reward += immediate_reward
        penalty = 0.5
        done = self._termination()
        actual_reward = reward - penalty
        observation = self._observation()
        return observation, actual_reward, done, {'reward': reward, 'penalty': penalty}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return seed


if __name__ == '__main__':
    grid_size = 22
    env = ParamTestEnv(grid_size)
    env.reset()
    # zigzag pattern
    horizontal_move = 0
    up = True
    terminated = False
    obs = [0, 0]
    total_return = 0
    step_counter = 0
    while not terminated:
        current_pos = round(grid_size * obs[-1])
        if up:
            if current_pos % grid_size != grid_size - 2:
                obs, step_reward, terminated, info = env.step(1)
                step_counter += 1
            elif horizontal_move < 1:
                obs, step_reward, terminated, info = env.step(0)
                step_counter += 1
                horizontal_move += 1
            else:
                horizontal_move = 0
                step_reward = 0
                up = False
        else:
            if current_pos % grid_size != 1:
                obs, step_reward, terminated, info = env.step(3)
                step_counter += 1
            elif horizontal_move < 1:
                obs, step_reward, terminated, info = env.step(0)
                step_counter += 1
                horizontal_move += 1
            else:
                horizontal_move = 0
                step_reward = 0
                up = True
        print('OBS: {0}, REWARD: {1}'.format(obs, step_reward))
        total_return += step_reward
    print('In {0} steps get {1} rewards'.format(step_counter, total_return))
