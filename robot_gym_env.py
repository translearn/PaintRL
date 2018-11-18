import os
import math
import obj_surface_process.bullet_paint_wrapper as p
import numpy as np
import gym
from gym.utils import seeding
from robot import Robot


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
        self.robot = None
        self._part_id = None
        self._renders = renders
        self._urdf_root = urdf_root
        self._setup_bullet_params()
        self._load_environment()

    def _setup_bullet_params(self):
        if self._renders:
            cid = self.p.connect(self.p.SHARED_MEMORY)
            if cid < 0:
                self.p.connect(self.p.GUI)
                self.p.resetDebugVisualizerCamera(2.6, 150, -60, (0.0, -0.2, 0.5))
        else:
            self.p.connect(self.p.DIRECT)

            self.p.resetSimulation()
            self.p.setTimeStep(1 / 240)
            self.p.setPhysicsEngineParameter(numSolverIterations=150)

    def _load_environment(self):
        self.p.loadURDF('plane.urdf', (0, 0, -0.93), useFixedBase=True)
        self._part_id = self.p.loadURDF(os.path.join(self._urdf_root, 'urdf', 'painting', 'door.urdf'),
                                        (-0.5, -0.5, 0.5), useFixedBase=True)
        # robot_urdf_path = os.path.join(self._urdf_root, 'urdf', 'franka_description', 'robots', 'panda_arm.urdf')
        # self._robot = Franka(robot_urdf_path)
        self.robot = Robot('kuka_iiwa/model_free_base.urdf', pos=(0.2, -0.2, 0),
                           orn=p.getQuaternionFromEuler((0, 0, math.pi*3/2)))
        self.p.setGravity(0, 0, -10)

    def _termination(self):
        # FIXME: which sign can be treated as terminal signal? remember the robot path?
        return False

    def _augmented_observation(self):
        robot_status = self.robot.get_observation()
        # TODO: camera in hand necessary?
        camera_data = []
        return [robot_status, camera_data]

    def _reward(self):
        # TODO: define reward
        return 1

    def _generate_paint_beams(self, end_effector_pose, end_effector_orn, show_debug_lines=False):
        radius = 0.25
        resolution = 0.02
        target_ray_plane = 0.5
        ray_origin = []
        ray_dst = []
        i = j = -radius
        while i <= radius:
            while j <= radius:
                # Euclidean distance within the radius
                if math.sqrt(math.pow(abs(i), 2) + math.pow(abs(j), 2)) <= radius:
                    ray_origin.append(end_effector_pose)
                    dst_ori = [i, j, target_ray_plane]
                    dst_target, _ = self.p.multiplyTransforms(end_effector_pose, end_effector_orn, dst_ori,
                                                              (0, 0, 0, 1))
                    ray_dst.append(dst_target)
                    if show_debug_lines:
                        p.addUserDebugLine(end_effector_pose, dst_target, (0, 1, 0))
                j += resolution
            i += resolution
            j = -radius
        return ray_origin, ray_dst

    def _get_tcp_orn_norm(self, end_effector_pose, end_effector_orn):
        p_along_tcp, _ = self.p.multiplyTransforms(end_effector_pose, end_effector_orn, (0, 0, 1), (0, 0, 0, 1))
        vector = [b - a for a, b in zip(end_effector_pose, p_along_tcp)]
        norm = np.linalg.norm(vector, ord=1)
        norm_vector = [v / norm for v in vector]
        return norm_vector

    def _paint(self, end_effector_pose, end_effector_orn, show_debug_lines=False):
        beams = self._generate_paint_beams(end_effector_pose, end_effector_orn, show_debug_lines)
        results = self.p.rayTestBatch(*beams)
        points = [item[3] for item in results]
        orn_norm = self._get_tcp_orn_norm(end_effector_pose, end_effector_orn)
        self.p.paint(self._part_id, points, (1, 0, 0), orn_norm)

    def step(self, action):
        # Make sure that the step should be small enough, otherwise the paint won't be continuous
        end_effector_pose, end_effector_orn = self.robot.apply_action(action)
        self.p.stepSimulation()
        self._paint(end_effector_pose, end_effector_orn, show_debug_lines=False)
        done = self._termination()
        reward = self._reward()
        observation = self._augmented_observation()

        return observation, reward, done, {}

    def reset(self):
        self.robot.reset()

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
    env = RobotGymEnv(os.path.dirname(os.path.realpath(__file__)), renders=True)
    pos = (0.0, 0.0, 0.6)
    orn = (0, -1, 0, 1)
    act = p.calculateInverseKinematics(env.robot.robot_id, env.robot._end_effector_idx, pos, orn)
    # action = [0, 0, 0, 0.5*math.pi, 0, -math.pi*0.5*0.66, 0]
    for joint in range(7):
        env.p.resetJointState(env.robot.robot_id, joint, targetValue=act[joint])
    env.step(act)
    pass
