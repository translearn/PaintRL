import os
import pybullet as p
import pybullet_data
import gym
from gym.utils import seeding
from franka import Franka


class RobotGymEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }
    reward_range = (-1e4, 1e4)

    # TODO: define them correctly
    action_space = None
    observation_space = None

    def __init__(self, urdf_root, renders=False):
        self.p = p
        self._robot = None
        self._renders = renders
        self._urdf_root = urdf_root
        self._setup_bullet_params()
        self._load_environment()

    def _setup_bullet_params(self):
        if self._renders:
            cid = self.p.connect(self.p.SHARED_MEMORY)
            if cid < 0:
                self.p.connect(self.p.GUI)
                self.p.resetDebugVisualizerCamera(2.6, 180, -41, [0.0, -0.2, -0.33])
        else:
            self.p.connect(self.p.DIRECT)

            self.p.resetSimulation()
            self.p.setPhysicsEngineParameter(fixedTimeStep=1 / 240, numSolverIterations=150)

    def _load_environment(self):
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.loadURDF('plane.urdf', [0, 0, -0.93], useFixedBase=True)
        robot_urdf_path = os.path.join(self._urdf_root, 'urdf', 'franka_description', 'robots', 'panda_arm.urdf')
        self._robot = Franka(robot_urdf_path)
        self.p.setGravity(0, 0, -10)

    def _termination(self):
        # FIXME: which sign can be treated as terminal signal? remember the robot path?
        return False

    def _augmented_observation(self):
        robot_status = self._robot.get_observation()
        # TODO: camera in hand necessary?
        camera_data = []
        return [robot_status, camera_data]

    def _reward(self):
        # TODO: define reward
        return 1

    def step(self, action):
        self._robot.apply_action(action)
        done = self._termination()
        reward = self._reward()
        observation = self._augmented_observation()

        return observation, reward, done, {}

    def reset(self):
        self._robot.reset()

    def render(self, mode='human'):
        if mode == 'human':
            raise Exception('please set render parameter to true to see the result')
        else:
            # TODO: figure out the meaning here.
            pass

    def close(self):
        self.p.disconnect()

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return seed


if __name__ == '__main__':
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # franka_urdf_path = os.path.join(current_dir, 'urdf', 'franka_description', 'robots', 'panda_arm.urdf')
    # f = Franka(franka_urdf_path)
    # print(franka_urdf_path)
    a = RobotGymEnv(os.path.dirname(os.path.realpath(__file__)), renders=True)
    pass
