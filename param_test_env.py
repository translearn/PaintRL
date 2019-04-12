import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class Observation:

    def __init__(self, part):
        self._part = part

    def reset_counters(self):
        raise NotImplementedError

    def refresh_counters(self, i, j):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError


class NoObservation(Observation):

    def reset_counters(self):
        pass

    def refresh_counters(self, i, j):
        pass

    def get_observation(self):
        return np.array([])


class GridObservation(Observation):

    def __init__(self, part):
        Observation.__init__(self, part)
        self._init_counter = int(self._part.init_reward_counter / 4)
        self._counter_1 = self._counter_2 = self._counter_3 = self._counter_4 = self._init_counter
        self.reset_counters()

    def reset_counters(self):
        self._counter_1 = self._counter_2 = self._counter_3 = self._counter_4 = self._init_counter

    def refresh_counters(self, i, j):
        if 1 <= i <= 10:
            if 1 <= j <= 10:
                self._counter_1 -= 1
            elif 11 <= j <= 20:
                self._counter_2 -= 1
        elif 11 <= i <= 20:
            if 1 <= j <= 10:
                self._counter_3 -= 1
            elif 11 <= j <= 20:
                self._counter_4 -= 1

    def get_observation(self):
        part_1 = self._counter_1 / self._init_counter
        part_2 = self._counter_2 / self._init_counter
        part_3 = self._counter_3 / self._init_counter
        part_4 = self._counter_4 / self._init_counter
        return np.array([part_1, part_2, part_3, part_4])


class Grid10Observation(Observation):

    def reset_counters(self):
        pass

    def refresh_counters(self, i, j):
        pass

    def get_observation(self):
        max_counter = int(self._part.init_reward_counter / 100)
        obs = np.zeros((10, 10), dtype=np.float64)
        for pos in self._part.world:
            if pos[0] in (0, 21) or pos[1] in (0, 21):
                continue
            x, y = int(pos[0] / 2 + 0.5) - 1, int(pos[1] / 2 + 0.5) - 1
            obs[x][y] += self._part.world[pos] / max_counter
        return obs.reshape((10 ** 2,))


class SectionObservation(Observation):

    def reset_counters(self):
        pass

    def refresh_counters(self, i, j):
        pass

    def get_observation(self):
        x, y = self._part.get_current_pos()
        counter_1 = counter_2 = counter_3 = counter_4 = 0
        max_1 = max_2 = max_3 = max_4 = 0
        for i, j in self._part.world:
            if 0 < i <= x:
                if 0 < j <= y:
                    counter_1 += self._part.world[(i, j)]
                    max_1 += 1
                elif y < j < self._part.size - 1:
                    counter_2 += self._part.world[(i, j)]
                    max_2 += 1
            elif x < i < self._part.size - 1:
                if 0 < j <= y:
                    counter_3 += self._part.world[(i, j)]
                    max_3 += 1
                elif y < j < self._part.size - 1:
                    counter_4 += self._part.world[(i, j)]
                    max_4 += 1
        obs_1 = 0 if max_1 == 0 else counter_1 / max_1
        obs_2 = 0 if max_2 == 0 else counter_2 / max_2
        obs_3 = 0 if max_3 == 0 else counter_3 / max_3
        obs_4 = 0 if max_4 == 0 else counter_4 / max_4
        return np.asarray([obs_1, obs_2, obs_3, obs_4])


class ParamTestEnv(gym.Env):
    OBS_MODE = 'section'

    reward_range = (-1e3, 1e3)
    action_space = spaces.Discrete(4)
    if OBS_MODE == 'section':
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float64)
    elif OBS_MODE == 'simple':
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float64)
    else:
        spaces.Box(low=0.0, high=1.0, shape=(100,), dtype=np.float64)

    EPISODE_MAX_LENGTH = 400

    def __init__(self, size, train_mode=True):
        self.size = size
        self._mode = train_mode
        self.world = {}
        self.visit_table = {}
        self._i = 1
        self._j = 1
        self._reward_counter = 0
        self._step_counter = 0
        self._violated_wall = False
        for i in range(self.size):
            for j in range(self.size):
                self.visit_table[(i, j)] = 0
                if i in (0, self.size - 1) or j in (0, self.size - 1):
                    self.world[(i, j)] = 0
                else:
                    self.world[(i, j)] = 1
                    self._reward_counter += 1
        self._init_world = self.world.copy()
        self.init_reward_counter = self._reward_counter
        self._visualizer = Visualizer(self.size)
        if self.OBS_MODE == 'grid':
            self._obs_handler = Grid10Observation(self)
        elif self.OBS_MODE == 'section':
            self._obs_handler = SectionObservation(self)
        else:
            self._obs_handler = NoObservation(self)

    def get_current_pos(self):
        return self._i, self._j

    def reset(self):
        self._i = 1
        self._j = 1
        self.visit_table[(self._i, self._j)] += 1
        self._violated_wall = False
        self._reward_counter = self.init_reward_counter
        self._step_counter = 0
        self.world = self._init_world.copy()
        self._obs_handler.reset_counters()
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
        if self._i < 0 or self._i >= self.size or self._j < 0 or self._j >= self.size:
            self._i = self._clip_pos(self._i)
            self._j = self._clip_pos(self._j)
            self._violated_wall = True
            return immediate_reward
        self.visit_table[(self._i, self._j)] += 1
        return immediate_reward

    def _clip_pos(self, pos):
        if pos < 0:
            pos = 0
        if pos >= self.size:
            pos = self.size - 1
        return pos

    def _termination(self):
        if self._violated_wall or self._reward_counter <= 0 or self._step_counter >= self.EPISODE_MAX_LENGTH - 1:
            return True
        return False

    def _observation(self):
        obs = self._obs_handler.get_observation()
        i = self._i / self.size
        j = self._j / self.size
        return np.append(obs, [i, j])
        # return obs

    def _get_immediate_reward(self):
        if self.world[(self._i, self._j)] > 0:
            self.world[(self._i, self._j)] -= 1
            self._reward_counter -= 1
            self._obs_handler.refresh_counters(self._i, self._j)
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
        penalty = 0.2
        done = self._termination()
        actual_reward = reward - penalty
        observation = self._observation()
        if not self._mode:
            x, y = round(observation[-2] * self.size), round(observation[-1] * self.size)
            # complete_pos = list(np.append(observation[:-2], pos))
            print('STEP: {0} ACTION: {1} OBS: [{2}, {3}], REWARD: {4}'.format(self._step_counter, action,
                                                                              x, y, actual_reward))
            # self._visualizer.print_world_table(self.world)
            if done:
                self._visualizer.print_visit_table(self.visit_table)
        return observation, actual_reward, done, {'reward': reward, 'penalty': penalty}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return seed


class Visualizer:

    def __init__(self, size):
        self._size = size
        self._template = '{0:3}'
        for i in range(1, self._size):
            self._template += '|{' + str(i) + ':3}'

    def _print_table(self, table):
        print(self._template.format(*[str(i) for i in range(self._size)]))
        for i in range(self._size):
            values = []
            for j in range(self._size):
                values.append(table[(i, j)])
            print(self._template.format(*values))

    def print_visit_table(self, table):
        print('Visit Table: count of visit in each state')
        self._print_table(table)

    def print_world_table(self, table):
        print('World Table:')
        self._print_table(table)


def zigzag():
    grid_size = 22
    env = ParamTestEnv(grid_size, train_mode=False)
    env.reset()
    # zigzag pattern
    horizontal_move = 0
    up = True
    terminated = False
    state = [0, 0]
    total_return = 0
    step_counter = 0
    while not terminated:
        current_pos = round(grid_size * state[-1])
        if up:
            if current_pos % grid_size != grid_size - 2:
                state, step_reward, terminated, info = env.step(1)
                step_counter += 1
            elif horizontal_move < 1:
                state, step_reward, terminated, info = env.step(0)
                step_counter += 1
                horizontal_move += 1
            else:
                horizontal_move = 0
                step_reward = 0
                up = False
        else:
            if current_pos % grid_size != 1:
                state, step_reward, terminated, info = env.step(3)
                step_counter += 1
            elif horizontal_move < 1:
                state, step_reward, terminated, info = env.step(0)
                step_counter += 1
                horizontal_move += 1
            else:
                horizontal_move = 0
                step_reward = 0
                up = True
        # print('OBS: {0}, REWARD: {1}'.format(state, step_reward))
        total_return += step_reward
    print('In {0} steps get {1} rewards'.format(step_counter, total_return))


def spiral():
    grid_size = 22
    env = ParamTestEnv(grid_size, train_mode=False)
    env.reset()
    done = False
    total_return = 0
    step_counter = 0
    direction = 0
    strait_counter = 19
    current_counter = strait_counter
    use_len = 3
    while not done:
        current_counter -= 1
        obs, reward, done, info = env.step(direction % 4)
        if current_counter == 0:
            direction += 1
            use_len -= 1
            if use_len <= 0:
                use_len = 2
                strait_counter -= 1
            current_counter = strait_counter
        step_counter += 1
        total_return += reward
    print('In {0} steps get {1} rewards'.format(step_counter, total_return))


if __name__ == '__main__':
    zigzag()
