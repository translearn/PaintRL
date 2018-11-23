import os
import math
import numpy as np
import obj_surface_process.bullet_paint_wrapper as p
import gym
from gym.utils import seeding
from robot import Robot


class RobotGymEnv(gym.Env):
    # TODO: find out a way to figure them out automatically.
    TEXTURE_WIDTH = 240
    TEXTURE_HEIGHT = 240

    RENDER_HEIGHT = 720
    RENDER_WIDTH = 960

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }
    reward_range = (-1e5, 1e5)
    action_space = gym.spaces.Box(np.array((-1, -1)), np.array((1, 1)), dtype=np.float32)
    observation_space = gym.spaces.Dict({
        'pose': gym.spaces.Box(np.array((-1, -1, -1)), np.array((1, 1, 1)), dtype=np.float32),
        'image': gym.spaces.Box(0, 255, [TEXTURE_WIDTH, TEXTURE_HEIGHT, 3], dtype=np.uint8)
    })

    def __init__(self, urdf_root, renders=False):
        self.p = p
        self.robot = None
        self._part_id = None
        self._start_points = None
        self._renders = renders
        self._urdf_root = urdf_root
        self._last_status = 0
        self._paint_side = p.Side.front
        self._paint_color = (1, 0, 0)
        self._setup_bullet_params()
        self._load_environment()

    def _setup_bullet_params(self):
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
                p.resetDebugVisualizerCamera(2.6, 150, -60, (0.0, -0.2, 0.5))
        else:
            p.connect(p.DIRECT)
            p.resetSimulation()
            p.setTimeStep(1 / 240)
            p.setPhysicsEngineParameter(numSolverIterations=150)

    def _load_environment(self):
        p.loadURDF('plane.urdf', (0, 0, 0), useFixedBase=True)
        self._part_id = p.loadURDF(os.path.join(self._urdf_root, 'urdf', 'painting', 'door.urdf'),
                                   (-0.5, -0.5, 0.5), useFixedBase=True)
        # robot_urdf_path = os.path.join(self._urdf_root, 'urdf', 'franka_description', 'robots', 'panda_arm.urdf')
        # self._robot = Franka(robot_urdf_path)
        self._start_points = p.get_start_points(self._part_id, p.Side.front)
        self.robot = Robot('kuka_iiwa/model_free_base.urdf', pos=(0.2, -0.2, 0),
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
        p.stepSimulation()
        reward = self._reward()
        done = self._termination()
        observation = self._augmented_observation()

        return observation, reward, done, {}

    def reset(self):
        self.robot.reset(self._start_points[0])

    def render(self, mode='human'):
        if mode == 'human':
            raise Exception('please set render parameter to true to see the result')
        else:
            view_matrix = (0.5649663805961609, -0.7801607251167297, 0.2686304450035095, 0.0,
                           0.8251139521598816, 0.5341863036155701, -0.18393480777740479, 0.0,
                           0.0, 0.32556772232055664, 0.9455187320709229, 0.0,
                           -0.02615496516227722, 0.1871221363544464, -1.1670949459075928, 1.0)

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
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # franka_urdf_path = os.path.join(current_dir, 'urdf', 'franka_description', 'robots', 'panda_arm.urdf')
    # f = Franka(franka_urdf_path)
    # print(franka_urdf_path)
    env = RobotGymEnv(os.path.dirname(os.path.realpath(__file__)), renders=True)
    env.reset()
    # pos = (0.0, 0.0, 0.6)
    # orn = (0, -1, 0, 1)
    # act = p.calculateInverseKinematics(env.robot.robot_id, 6, pos, orn)
    # # action = [0, 0, 0, 0.5*math.pi, 0, -math.pi*0.5*0.66, 0]
    # for joint in range(7):
    #     env.p.resetJointState(env.robot.robot_id, joint, targetValue=act[joint])
    for _ in range(10):
        act = [0.6, 0.4]
        env.step(act)
    pass
