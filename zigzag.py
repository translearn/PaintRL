import os
from PaintRLEnv.robot_gym_env import RobotGymEnv

EXTRA_CONFIG = {
    'RENDER_HEIGHT': 720,
    'RENDER_WIDTH': 960,

    'Part_NO': 1,
    'Expected_Episode_Length': 245,
    'EPISODE_MAX_LENGTH': 245,

    'TERMINATION_MODE': 'late',
    'SWITCH_THRESHOLD': 0.9,

    'START_POINT_MODE': 'fixed',
    'TURNING_PENALTY': False,
    'OVERLAP_PENALTY': False,
    'COLOR_MODE': 'RGB',
}


def simple_rgb_zigzag():
    RobotGymEnv.change_action_mode(1, 'discrete', 4)
    RobotGymEnv.change_obs_mode('discrete', 4)

    with RobotGymEnv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv'), with_robot=False,
                     renders=True, render_video=False, rollout=True, extra_config=EXTRA_CONFIG) as env:
        horizontal_move = 0
        up = True
        done = False
        obs = [0] * 5
        total_return = 0
        step_counter = 0
        while not done:
            current_pos = 0 if obs[-1] == 0 else round(1 / obs[-1])
            if up:
                if current_pos % 22 != 19:
                    obs, reward, done, info = env.step(1)
                    step_counter += 1
                elif horizontal_move < 2:
                    obs, reward, done, info = env.step(0)
                    step_counter += 1
                    horizontal_move += 1
                else:
                    horizontal_move = 0
                    reward = 0
                    up = False
            else:
                if current_pos % 22 != 2:
                    obs, reward, done, info = env.step(3)
                    step_counter += 1
                elif horizontal_move < 2:
                    obs, reward, done, info = env.step(0)
                    step_counter += 1
                    horizontal_move += 1
                else:
                    horizontal_move = 0
                    reward = 0
                    up = True
            print('OBS: {0}, REWARD: {1}'.format(obs, reward))
            total_return += reward
        print('In {0} steps get {1} rewards'.format(step_counter, total_return))


def simple_rgb1_zigzag():
    RobotGymEnv.change_action_mode(1, 'discrete', 4)
    RobotGymEnv.change_obs_mode('simple', 4)

    with RobotGymEnv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv'), with_robot=False,
                     renders=True, render_video=False, rollout=True, extra_config=EXTRA_CONFIG) as env:
        horizontal_move = 0
        up = True
        done = False
        obs = [0, 0]
        total_return = 0
        step_counter = 0
        while not done:
            if up:
                if obs[1] < 0.95:
                    obs, reward, done, info = env.step(1)
                    step_counter += 1
                elif horizontal_move < 2:
                    obs, reward, done, info = env.step(0)
                    step_counter += 1
                    horizontal_move += 1
                else:
                    horizontal_move = 0
                    reward = 0
                    up = False
            else:
                if obs[1] > 0.05:
                    obs, reward, done, info = env.step(3)
                    step_counter += 1
                elif horizontal_move < 2:
                    obs, reward, done, info = env.step(0)
                    step_counter += 1
                    horizontal_move += 1
                else:
                    horizontal_move = 0
                    reward = 0
                    up = True
            print('OBS: {0}, REWARD: {1}'.format(obs, reward))
            total_return += reward
        print('In {0} steps get {1} rewards'.format(step_counter, total_return))


def simple_rgb_profile_zigzag():
    RobotGymEnv.change_action_mode(1, 'discrete', 4)
    RobotGymEnv.change_obs_mode('grid', 10)

    with RobotGymEnv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv'), with_robot=False,
                     renders=False, render_video=False, rollout=False, extra_config=EXTRA_CONFIG) as env:
        for _ in range(100):
            env.reset()
            horizontal_move = 0
            up = True
            done = False
            obs = [0, 0]
            total_return = 0
            step_counter = 0
            while not done:
                if up:
                    if obs[1] < 0.95:
                        obs, reward, done, info = env.step(1)
                        step_counter += 1
                    elif horizontal_move < 2:
                        obs, reward, done, info = env.step(0)
                        step_counter += 1
                        horizontal_move += 1
                    else:
                        horizontal_move = 0
                        reward = 0
                        up = False
                else:
                    if obs[1] > 0.05:
                        obs, reward, done, info = env.step(3)
                        step_counter += 1
                    elif horizontal_move < 2:
                        obs, reward, done, info = env.step(0)
                        step_counter += 1
                        horizontal_move += 1
                    else:
                        horizontal_move = 0
                        reward = 0
                        up = True
                print('OBS: {0}, REWARD: {1}'.format(obs, reward))
                total_return += reward
            print('In {0} steps get {1} rewards'.format(step_counter, total_return))


def simple_hsi_zigzag():
    EXTRA_CONFIG['COLOR_MODE'] = 'HSI'
    RobotGymEnv.change_action_mode(2, 'continuous')
    RobotGymEnv.change_obs_mode('simple', 4)

    with RobotGymEnv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv'), with_robot=False,
                     renders=True, render_video=False, rollout=True, extra_config=EXTRA_CONFIG) as env:
        horizontal_move = 0
        up = True
        done = False
        obs = [0, 0]
        total_return = 0
        step_counter = 0
        while not done:
            if up:
                if obs[1] < 0.95:
                    obs, reward, done, info = env.step([0, 1])
                    step_counter += 1
                elif horizontal_move < 2:
                    obs, reward, done, info = env.step([0.5, 0])
                    step_counter += 1
                    horizontal_move += 1
                else:
                    horizontal_move = 0
                    reward = 0
                    up = False
            else:
                if obs[1] > 0.05:
                    obs, reward, done, info = env.step([0, -1])
                    step_counter += 1
                elif horizontal_move < 2:
                    obs, reward, done, info = env.step([0.5, 0])
                    step_counter += 1
                    horizontal_move += 1
                else:
                    horizontal_move = 0
                    reward = 0
                    up = True
            print('OBS: {0}, REWARD: {1}'.format(obs, reward))
            total_return += reward
        print('In {0} steps get {1} rewards'.format(step_counter, total_return))


def profile_rgb_zigzag():
    import cProfile as Profile

    pr = Profile.Profile()
    pr.disable()
    pr.enable()
    simple_rgb_profile_zigzag()
    pr.disable()
    pr.dump_stats('/home/pyang/profile_grid.pstat')


def show_profile_result():
    import pstats

    ps = pstats.Stats('/home/pyang/profile.pstat')
    ps.strip_dirs().print_stats()


if __name__ == '__main__':
    # simple_rgb1_zigzag()
    simple_hsi_zigzag()
    # profile_rgb_zigzag()
