import os
from PaintRLEnv.robot_gym_env import RobotGymEnv


def simple_rgb_spiral():
    RobotGymEnv.set_start_point_mode('all')
    RobotGymEnv.change_action_mode(1, 'discrete', 4)
    RobotGymEnv.change_obs_mode('simple', 4)
    RobotGymEnv.switch_turning_penalty(True)
    RobotGymEnv.set_termination_mode('late')
    RobotGymEnv.COLOR_MODE = 'RGB'
    with RobotGymEnv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv'), with_robot=False,
                     renders=True, render_video=False, rollout=True) as env:
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
            total_return += reward
        print('In {0} steps get {1} rewards'.format(step_counter, total_return))


def simple_hsi_spiral():
    RobotGymEnv.set_start_point_mode('all')
    RobotGymEnv.change_action_mode(1, 'discrete', 4)
    RobotGymEnv.change_obs_mode('simple', 4)
    RobotGymEnv.switch_turning_penalty(True)
    RobotGymEnv.set_termination_mode('late')
    RobotGymEnv.COLOR_MODE = 'HSI'
    with RobotGymEnv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv'), with_robot=False,
                     renders=True, render_video=False, rollout=True) as env:
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
            total_return += reward
        print('In {0} steps get {1} rewards'.format(step_counter, total_return))


if __name__ == '__main__':
    simple_hsi_spiral()
