import os
from PaintRLEnv.robot_gym_env import PaintGymEnv

EXTRA_CONFIG = {
    'RENDER_HEIGHT': 720,
    'RENDER_WIDTH': 960,

    'Part_NO': 1,
    'Expected_Episode_Length': 245,
    'EPISODE_MAX_LENGTH': 245,

    'TERMINATION_MODE': 'late',
    'SWITCH_THRESHOLD': 0.9,

    'START_POINT_MODE': 'all',
    'TURNING_PENALTY': False,
    'OVERLAP_PENALTY': False,
    'COLOR_MODE': 'RGB',
}


def simple_rgb_spiral():
    PaintGymEnv.change_action_mode(1, 'discrete', 4)
    PaintGymEnv.change_obs_mode('simple', 4)

    with PaintGymEnv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv'), with_robot=False,
                     renders=True, render_video=False, rollout=True, extra_config=EXTRA_CONFIG) as env:
        start_points = getattr(env, '_start_points')
        axis_1, axis_2 = [], []
        for sp in start_points:
            axis_1.append(sp[0][1])
            axis_2.append(sp[0][2])

        x = min(axis_1) + (max(axis_1) - min(axis_1)) / 2
        y = min(axis_2) + (max(axis_2) - min(axis_2)) / 2

        center_point = [[start_points[0][0][0], x, y], start_points[0][1]]
        env.robot.reset(center_point)
        done = False
        total_return = 0
        step_counter = 0
        direction = 0
        strait_counter = 1
        current_counter = strait_counter
        while not done:
            current_counter -= 1
            obs, reward, done, info = env.step(direction % 4)
            if current_counter == 0:
                strait_counter += 1
                direction += 1
                current_counter = strait_counter
            step_counter += 1
            print('OBS: {0}, REWARD: {1}'.format(obs, reward))
            total_return += reward
        print('In {0} steps get {1} rewards'.format(step_counter, total_return))


def simple_hsi_spiral():
    EXTRA_CONFIG['COLOR_MODE'] = 'HSI'
    PaintGymEnv.change_action_mode(1, 'discrete', 4)
    PaintGymEnv.change_obs_mode('simple', 4)

    with PaintGymEnv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv'), with_robot=False,
                     renders=True, render_video=False, rollout=True, extra_config=EXTRA_CONFIG) as env:
        start_points = getattr(env, '_start_points')
        axis_1, axis_2 = [], []
        for sp in start_points:
            axis_1.append(sp[0][1])
            axis_2.append(sp[0][2])

        x = min(axis_1) + (max(axis_1) - min(axis_1)) / 2
        y = min(axis_2) + (max(axis_2) - min(axis_2)) / 2

        center_point = [[start_points[0][0][0], x, y], start_points[0][1]]
        env.robot.reset(center_point)
        done = False
        total_return = 0
        step_counter = 0
        direction = 0
        strait_counter = 1
        current_counter = strait_counter
        while not done:
            current_counter -= 1
            obs, reward, done, info = env.step(direction % 4)
            if current_counter == 0:
                strait_counter += 1
                direction += 1
                current_counter = strait_counter
            step_counter += 1
            print('OBS: {0}, REWARD: {1}'.format(obs, reward))
            total_return += reward
        print('In {0} steps get {1} rewards'.format(step_counter, total_return))


if __name__ == '__main__':
    simple_rgb_spiral()
