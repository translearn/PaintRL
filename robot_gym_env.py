import os
import math
import time
import numpy as np
import obj_surface_process.bullet_paint_wrapper as p
import pybullet_data
import gym
from gym.utils import seeding
from robot import Robot
from video_renderer import VideoRecorder
from timeit import default_timer as timer


def _get_view_matrix():
    cam_target_pos = (-0.03, -0.25, 0.82)
    cam_distance = 1
    pitch = -33.60
    yaw = 52.00
    roll = 0
    up_axis_index = 2
    return p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, yaw, pitch, roll, up_axis_index)


class StepManager:
    """
    Manage video record function, a use case of dependency injection pattern to decouple the dependency between
    RobotGymEnv and Robot
    """
    TIME_STEP = 1 / 240

    def __init__(self, gym_env, render_video=False, video_dir=None):
        self._env = gym_env
        self._render_video = render_video
        self._video_recorder = None
        self._last_time = None
        self._step_counter = 0
        self._video_dir = video_dir
        self._episode_counter = 0
        self._steps_per_frame = int(1 / (self._env.metadata.get('video.frames_per_second', 30) * StepManager.TIME_STEP))
        if self._render_video:
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
        if self._render_video:
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


class RobotGymEnv(gym.Env):

    RENDER_HEIGHT = 720
    RENDER_WIDTH = 960

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}
    reward_range = (-1e5, 1e5)
    action_space = gym.spaces.Box(np.array((-1, -1)), np.array((1, 1)), dtype=np.float32)
    # will be setup after the part loaded and the size of the texture is clear
    observation_space = None

    def __init__(self, urdf_root, renders=False, render_video=False):
        self.robot = None
        self._part_id = None
        self._start_points = None
        self._renders = renders
        self._render_video = render_video
        self._urdf_root = urdf_root

        self._last_status = 0
        self._paint_side = p.Side.front
        self._paint_color = (1, 0, 0)

        self._setup_bullet_params()
        self._step_manager = StepManager(self, render_video, '/home/pyang/Videos/')
        self._load_environment()

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._render_video:
            self._step_manager.close_video_recorder()
        self.close()

    def _setup_bullet_params(self):
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
                p.resetDebugVisualizerCamera(1.40, 52.00, -33.60, (0.0, -0.2, 0.5))
        else:
            p.connect(p.DIRECT)
            p.resetSimulation()
            p.setTimeStep(1 / 240)
            p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _load_environment(self):
        p.loadURDF('plane.urdf', (0, 0, 0), useFixedBase=True)
        self._part_id = p.load_part(os.path.join(self._urdf_root, 'urdf', 'painting', 'door.urdf'),
                                    (-0.4, -0.6, 0.3), useFixedBase=True)
        texture_width, texture_height = p.get_texture_size(self._part_id)
        RobotGymEnv.observation_space = gym.spaces.Dict({
            'pose': gym.spaces.Box(np.array((-1, -1, -1)), np.array((1, 1, 1)), dtype=np.float32),
            'image': gym.spaces.Box(0, 255, [texture_width, texture_height, 3], dtype=np.uint8)})
        self._start_points = p.get_start_points(self._part_id, p.Side.front)
        self.robot = Robot(self._step_manager, 'kuka_iiwa/model_free_base.urdf', pos=(0.2, -0.2, 0),
                           orn=p.getQuaternionFromEuler((0, 0, math.pi*3/2)))
        p.setGravity(0, 0, -10)

    def _termination(self):
        max_possible_point = p.get_job_limit(self._part_id, self._paint_side)
        return False if max_possible_point > self._last_status else True

    def _augmented_observation(self):
        observation = {}
        pose, orn_norm = self.robot.get_observation()
        image = p.get_texture_image(self._part_id)
        # image.show()
        observation['pose'] = pose
        observation['image'] = image
        return observation

    def _reward(self):
        current_status = p.get_job_status(self._part_id, self._paint_side, self._paint_color)
        reward = current_status - self._last_status
        self._last_status = current_status
        return reward

    def step(self, action):
        self.robot.apply_action(action, self._part_id, self._paint_color)
        reward = self._reward()
        done = self._termination()
        observation = self._augmented_observation()

        return observation, reward, done, {}

    def reset(self):
        # p.addUserDebugLine((0, 0, 0), self._start_points[0][0], (0, 1, 0))
        self.robot.reset(self._start_points[0])

    def render(self, mode='human'):
        if mode == 'human':
            raise Exception('please set render parameter to true to see the result')
        else:
            # call _get_view_matrix to get it, write out explicitly to speed up render process
            view_matrix = (0.6156615018844604, -0.4360785186290741, 0.6563509702682495, 0.0,
                           0.788010835647583, 0.3407019078731537, -0.5127975344657898, 0.0,
                           -4.470348358154297e-08, 0.8329213857650757, 0.5533915758132935, 0.0,
                           0.21547260880470276, -0.6109024286270142, -1.5622899532318115, 1.0)

            proj_matrix = (0.7499999403953552, 0.0, 0.0, 0.0,
                           0.0, 1.0, 0.0, 0.0,
                           0.0, 0.0, -1.0000200271606445, -1.0,
                           0.0, 0.0, -0.02000020071864128, 0.0)

            _, _, px, _, _ = p.getCameraImage(width=RobotGymEnv.RENDER_WIDTH, height=RobotGymEnv.RENDER_HEIGHT,
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
    with RobotGymEnv(os.path.dirname(os.path.realpath(__file__)), renders=True, render_video=False) as env:
        for _ in range(7):
            env.step([0, 1])
        env.step([1, 0])
        pass
        # for _ in range(10):
        #     # act = [random.random(), random.random()]
        #     act = [0.6, 0.4]
        #     env.step(act)
