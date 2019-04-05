import os
from PaintRLEnv.robot_gym_env import RobotGymEnv

RobotGymEnv.set_start_point_mode('fixed')
RobotGymEnv.change_action_mode(1, 'discrete', 4)
RobotGymEnv.change_obs_mode('simple', 4)
RobotGymEnv.switch_turning_penalty(True)
RobotGymEnv.set_termination_mode('late')
RobotGymEnv.COLOR_MODE = 'HSI'


if __name__ == '__main__':
    with RobotGymEnv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv'), with_robot=False,
                     renders=True, render_video=False, rollout=True) as env:
        horizontal_move = 0
        up = True
        done = False
        obs = [0, 0]
        total_return = 0
        step_counter = 0
        while not done:
            step_counter += 1
            if up:
                if obs[1] < 0.95:
                    obs, reward, done, info = env.step(1)
                elif horizontal_move < 2:
                    obs, reward, done, info = env.step(0)
                    horizontal_move += 1
                else:
                    horizontal_move = 0
                    reward = 0
                    up = False
            else:
                if obs[1] > 0.05:
                    obs, reward, done, info = env.step(3)
                elif horizontal_move < 2:
                    obs, reward, done, info = env.step(0)
                    horizontal_move += 1
                else:
                    horizontal_move = 0
                    reward = 0
                    up = True
            total_return += reward
        print('In {0} steps get {1} rewards'.format(step_counter, total_return))
