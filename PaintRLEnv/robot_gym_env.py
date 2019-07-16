import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import math
import time
from random import randint
import numpy as np
import obj_surface_process.bullet_paint_wrapper as p
import pybullet_data
import gym
from gym import spaces
from gym.utils import seeding
from robot import Robot
from video_renderer import VideoRecorder
from timeit import default_timer as timer


def _get_view_matrix():
    # Change the values, calculate the new matrix, then paste it in the render function of the env class.
    cam_target_pos = (-0.03, -0.25, 0.82)
    cam_distance = 1
    pitch = -33.60
    yaw = 52.00
    roll = 0
    up_axis_index = 2
    return p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, yaw, pitch, roll, up_axis_index)


class StepManager:
    """
    Manage video record function, using dependency injection pattern to decouple the dependency between
    RobotGymEnv and Robot
    """
    TIME_STEP = 1 / 240

    def __init__(self, gym_env, render_video=False, video_dir=None):
        self._env = gym_env
        self.render_video = render_video
        self._video_recorder = None
        self._last_time = None
        self._step_counter = 0
        self._video_dir = video_dir
        self._episode_counter = 0
        self._steps_per_frame = int(1 / (self._env.metadata.get('video.frames_per_second', 30) * StepManager.TIME_STEP))
        if self.render_video:
            self.reset_video_recorder()
        self._reset_counters()

    def _reset_counters(self):
        self._step_counter = 0
        self._last_time = timer()

    def _separate_frame(self):
        now = timer()
        calculation_time = now - self._last_time
        if calculation_time < StepManager.TIME_STEP:
            time.sleep(StepManager.TIME_STEP - calculation_time)
        self._last_time = now

    def _capture_frame(self):
        if self._step_counter == 0:
            self._step_counter = self._steps_per_frame
            self._video_recorder.capture_frame()
        else:
            self._step_counter = self._step_counter - 1

    def step_simulation(self):
        p.stepSimulation()
        if self.render_video:
            self._separate_frame()
            self._capture_frame()

    def reset_video_recorder(self):
        if self._video_recorder:
            self.close_video_recorder()
        base_path = os.path.join(self._video_dir, 'video_episode{0}'.format(self._episode_counter))
        metadata = {'episode_id': self._episode_counter}
        self._video_recorder = VideoRecorder(env=self._env, base_path=base_path, metadata=metadata, enabled=True)
        self._reset_counters()

    def close_video_recorder(self, video_info=()):
        self._video_recorder.capture_frame()
        video_path = self._video_recorder.get_path()
        self._video_recorder.close()
        if video_info:
            file_extension_position = video_path.find(".", len(video_path) - 5)
            file_name_old = video_path[:file_extension_position]
            file_extension = video_path[file_extension_position:]
            file_name_new = file_name_old + "_" + video_info + file_extension
            os.rename(video_path, file_name_new)
        self._video_recorder = None


def _handle_pos(pos):
    if pos == 0:
        return 0
    elif pos == 1:
        return 21
    else:
        return int(pos * 20) + 1


def _get_discrete_obs(obs):
    x, y = _handle_pos(obs[0]), _handle_pos(obs[1])
    return (x + 1) * 22 + y


Part_Dict = {
    0: ['door_test.urdf', 9148],
    1: ['square.urdf', 14350],
    2: ['door_lf.urdf', 0],
    3: ['door_lr.urdf', 0],
    4: ['door_rf.urdf', 0],
    5: ['door_rr.urdf', 17000],
    6: ['roof.urdf', 0],
    7: ['bonnet.urdf', 0],
    8: ['door_rr_big.urdf', 0],
    9: ['test.urdf', 9148],
}


class RobotGymEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    reward_range = (-1e3, 1e3)

    # Adjust env by hand when using Ray!!!
    ACTION_SHAPE = 1
    ACTION_MODE = 'discrete'
    DISCRETE_GRANULARITY = 4

    OBS_MODE = 'section'
    OBS_GRAD = 4

    EXTRA_CONFIG = {
        'RENDER_HEIGHT': 720,
        'RENDER_WIDTH': 960,

        'Part_NO': 0,
        'Expected_Episode_Length': 245,
        'EPISODE_MAX_LENGTH': 245,

        # 'early', termination controlled by average reward
        # 'late', termination clipped by max permitted step
        # 'hybrid', termination is early at first, after reached threshold will switch to late mode
        'TERMINATION_MODE': 'late',
        # Switch theshold in hybrid mode
        'SWITCH_THRESHOLD': 0.9,

        # 'fixed' only one point,
        # 'anchor' four anchor points,
        # 'edge' edge points,
        # 'all' all points, namely all of the triangle centers
        'START_POINT_MODE': 'anchor',
        'TURNING_PENALTY': False,
        'OVERLAP_PENALTY': False,
        'COLOR_MODE': 'RGB',
    }

    if ACTION_MODE == 'continuous':
        if ACTION_SHAPE == 2:
            action_space = spaces.Box(np.array((-1, -1)), np.array((1, 1)), dtype=np.float64)
        else:
            action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
    else:
        action_space = spaces.Discrete(DISCRETE_GRANULARITY)
    if OBS_MODE == 'section':
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_GRAD + 2,), dtype=np.float64)
    elif OBS_MODE == 'grid':
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_GRAD ** 2,), dtype=np.float64)
    elif OBS_MODE == 'simple':
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float64)
    else:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_GRAD + 1,), dtype=np.float64)

    # class methods below does not support in ray distributed framework
    @classmethod
    def change_obs_mode(cls, mode='section', grad=5):
        """
        change the observation mode
        :param mode: 'section', 'grid', 'simple', 'discrete'
        :param grad: grad of the observation
        :return:
        """
        cls.OBS_MODE = mode
        if mode == 'section':
            cls.observation_space = spaces.Box(low=0.0, high=1.0, shape=(18 + 2,), dtype=np.float64)
        elif mode == 'grid':
            cls.observation_space = spaces.Box(low=0.0, high=1.0, shape=(cls.OBS_GRAD ** 2,), dtype=np.float64)
        elif mode == 'simple':
            cls.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float64)
        else:
            cls.observation_space = spaces.Box(low=0.0, high=1.0, shape=(cls.OBS_GRAD + 1,), dtype=np.float64)
        cls.OBS_GRAD = grad

    @classmethod
    def change_action_mode(cls, shape=2, mode='continuous', discrete_granularity=20):
        cls.ACTION_SHAPE = shape
        cls.ACTION_MODE = mode
        if mode == 'continuous':
            if shape == 1:
                cls.action_space = spaces.Box(np.array(-1,), np.array(-1,), dtype=np.float64)
            else:
                cls.action_space = spaces.Box(np.array((-1, -1)), np.array((1, 1)), dtype=np.float64)
        else:
            cls.action_space = spaces.Discrete(discrete_granularity)

    def __init__(self, urdf_root, with_robot=True, renders=False, render_video=False,
                 rollout=False, extra_config=None):
        if extra_config is None:
            extra_config = self.EXTRA_CONFIG
        self._setup_extra_config(extra_config)
        self._with_robot = with_robot
        self._renders = renders
        self._render_video = render_video
        self._urdf_root = urdf_root
        self._rollout = rollout

        # self._last_status = 0
        self._step_counter = 0
        self._total_reward = 0
        self._total_return = 0
        self._paint_side = p.Side.front
        # Monotone, multi-color should not be used
        self._paint_color = (1, 0, 0)

        self._setup_bullet_params()
        self._step_manager = StepManager(self, render_video, '/home/pyang/Videos/')
        self._load_environment()
        self.replay_buffer = []

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._render_video:
            self._step_manager.close_video_recorder()
        self.close()

    def _setup_extra_config(self, config):
        self.RENDER_WIDTH = config['RENDER_WIDTH']
        self.RENDER_HEIGHT = config['RENDER_HEIGHT']
        self._part_name = Part_Dict[config['Part_NO']][0]
        self._max_possible_point = Part_Dict[config['Part_NO']][1]
        self.Expected_Episode_Length = config['Expected_Episode_Length']
        self.EPISODE_MAX_LENGTH = config['EPISODE_MAX_LENGTH']
        self.TERMINATION_MODE = config['TERMINATION_MODE']
        self.SWITCH_THRESHOLD = config['SWITCH_THRESHOLD']
        self.START_POINT_MODE = config['START_POINT_MODE']
        self.TURNING_PENALTY = config['TURNING_PENALTY']
        self.OVERLAP_PENALTY = config['OVERLAP_PENALTY']
        self.COLOR_MODE = config['COLOR_MODE']

    def _setup_bullet_params(self):
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
                # p.resetDebugVisualizerCamera(1.40, 52.00, -33.60, (0.0, -0.2, 0.5))
                # sheet screenshot angle
                p.resetDebugVisualizerCamera(1.0, 90.40, -3.20, (0.00, -0.18, 0.73))
                # door screenshot angle
                # p.resetDebugVisualizerCamera(1.0, 90.40, -3.20, (-0.60, -0.19, 0.65))
        else:
            p.connect(p.DIRECT)
            p.resetSimulation()
            p.setTimeStep(1 / 240)
            p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _load_environment(self):
        if self._renders:
            p.loadURDF('plane.urdf', (0, 0, 0), useFixedBase=True)
        path = os.path.join(self._urdf_root, 'urdf', 'painting', self._part_name)
        self._part_id = p.loadURDF(path, (-0.4, -0.6, 0.25), useFixedBase=True, flags=p.URDF_ENABLE_SLEEPING)
        p.load_part(self._part_id, self._renders, self.OBS_MODE, self.OBS_GRAD, self.COLOR_MODE, path,
                    self._paint_side, self._paint_color)

        self._start_points = p.get_start_points(self._part_id, mode=self.START_POINT_MODE)
        # Switch on the capture texture function manually here.
        self.robot = Robot(self._step_manager, 'kuka_iiwa/model_free_base.urdf', pos=(0.2, -0.2, 0),
                           orn=p.getQuaternionFromEuler((0, 0, math.pi*3/2)),
                           with_robot=self._with_robot, capture_texture=False)
        density = p.get_side_density(self._part_id)
        self.robot.set_up_paint_params(self.COLOR_MODE, density)
        p.setGravity(0, 0, -10)
        self.reset()

    def _termination(self):
        self._step_counter += 1
        # self._max_possible_point = p.get_job_limit(self._part_id)
        finished = False if self._max_possible_point > self._total_reward * 100 else True
        robot_termination = self.robot.termination_request()

        avg_reward = self._total_reward / self._step_counter
        # switch the mode of termination
        expected_avg_reward = self._max_possible_point / (self.Expected_Episode_Length * 100)
        if avg_reward < expected_avg_reward and self.TERMINATION_MODE != 'late':
            if self.TERMINATION_MODE == 'early':
                return True
            # Hybrid mode
            elif self._total_reward < self.SWITCH_THRESHOLD * self._max_possible_point / 100:
                return True
        return finished or robot_termination or self._step_counter > self.EPISODE_MAX_LENGTH - 1

    def _augmented_observation(self):
        pose, _ = self.robot.get_observation()
        normalized_pose = p.get_normalized_pose(self._part_id, pose)
        if self.OBS_MODE == 'simple':
            return list(normalized_pose)
        status = p.get_observation(self._part_id, pose)
        if self.OBS_MODE == 'grid':
            return status
        elif self.OBS_MODE == 'discrete':
            position = _get_discrete_obs(normalized_pose)
            obs = list(status)
            obs.append(np.float64(1 / position))
            return obs
        return list(status) + list(normalized_pose)

    def _reward(self, succeeded_counter):
        # Rescale the reward
        reward = succeeded_counter / 100
        self._total_reward += reward
        return reward

    def _penalty(self, paint_succeed_rate):
        # time_step_penalty = 0.1
        time_step_penalty = 0.2
        # off_part_penalty = self.robot.off_part_penalty
        # total_penalty = time_step_penalty + off_part_penalty
        total_penalty = time_step_penalty
        if self.OVERLAP_PENALTY:
            overlap_penalty = 0.1 * (1 - paint_succeed_rate)
            # overlap_penalty = 1 - paint_succeed_rate
            total_penalty += overlap_penalty
        if self.TURNING_PENALTY:
            turning_penalty = 0.1 * (self.robot.get_angle_diff() / math.pi)
            total_penalty += turning_penalty
        return total_penalty

    def _preprocess_action(self, action):
        if self.ACTION_MODE == 'continuous':
            return action
        else:
            action = action - self.action_space.n / 2
            return [2 * action / self.action_space.n]

    def step(self, action):
        p_action = self._preprocess_action(action)
        paint_succeed_rate, succeeded_counter = self.robot.apply_action(p_action, self._part_id)
        reward = self._reward(succeeded_counter)
        penalty = self._penalty(paint_succeed_rate)
        actual_reward = reward - penalty
        done = self._termination()
        # if paint_succeed_rate < 0.5:
        #     done = True
        observation = self._augmented_observation()
        if not done:
            self._total_return += actual_reward
        if self._renders:
            p.write_text_info(self._part_id, action, reward, penalty, self._total_return, self._step_counter)
            if self._rollout:
                self.replay_buffer.append(action)
                if done:
                    # Store the rollout results and print out.
                    print(self.replay_buffer)
        return observation, actual_reward, done,  {'reward': reward, 'penalty': penalty}

    def reset(self):
        if self._rollout:
            p.removeAllUserDebugItems()
            p.reset_part(self._part_id, 0, 0)
            start_point = self._start_points[0]
            self.replay_buffer = []
        else:
            painted_percent = 0  # randint(0, 49)
            painted_mode = randint(0, 7)
            # start_point = p.reset_part(self._part_id, painted_percent, painted_mode, with_start_point=True)
            p.reset_part(self._part_id, painted_percent, painted_mode, with_start_point=False)
            start_point = self._start_points[randint(0, len(self._start_points) - 1)]
        self._step_counter = 0
        self._total_return = 0
        self._total_reward = 0
        self.robot.reset(start_point)
        # self._last_status = p.get_job_status(self._part_id, self._paint_side, self._paint_color)
        return self._augmented_observation()

    def render(self, mode='human'):
        if mode == 'human':
            raise Exception('please set render parameter to true to see the result')
        else:
            # call _get_view_matrix to get it, write out explicitly to speed up render process
            # Left side View
            # view_matrix = (0.6156615018844604, -0.4360785186290741, 0.6563509702682495, 0.0,
            #                0.788010835647583, 0.3407019078731537, -0.5127975344657898, 0.0,
            #                -4.470348358154297e-08, 0.8329213857650757, 0.5533915758132935, 0.0,
            #                0.21547260880470276, -0.6109024286270142, -1.5622899532318115, 1.0)
            # Front view
            view_matrix = (-0.006981172598898411, -0.05582012981176376, 0.998416543006897, 0.0,
                           0.9999756813049316, -0.00038970436435192823, 0.006970287300646305, 0.0,
                           4.94765073355552e-09, 0.9984409213066101, 0.055821485817432404, 0.0,
                           0.18580667674541473, -0.682552695274353, -0.4359097480773926, 1.0)

            proj_matrix = (0.7499999403953552, 0.0, 0.0, 0.0,
                           0.0, 1.0, 0.0, 0.0,
                           0.0, 0.0, -1.0000200271606445, -1.0,
                           0.0, 0.0, -0.02000020071864128, 0.0)

            _, _, px, _, _ = p.getCameraImage(width=self.RENDER_WIDTH, height=self.RENDER_HEIGHT,
                                              viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)
            rgb_array = np.array(px)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    def close(self):
        p.disconnect()

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return seed


if __name__ == '__main__':
    with RobotGymEnv(os.path.dirname(os.path.realpath(__file__)), with_robot=False,
                     renders=True, render_video=False, rollout=True) as env:
        # i = 0
        # while i <= 1:
        #     env.step([i, 1])
        #     env.step([-i, -1])
        #     i += 0.01
        # exit()
        for i in range(8):
            env.step(i)
            print(env.robot.get_angle_diff())
            env.step(8 - i)
        env.reset()
        # env.step([1, 1])
        # for _ in range(20):
        #     env.step([0, 1])
        # env.reset()
        # env.step([0, 1])
        # env.step([0, 1])

        # Paste the actions in the replay buffer into the list below.
        # replay_buf = []
        # for a in replay_buf:
        #     env.step(a)
