import os
import tensorflow as tf
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import flatten
from ray.tune.registry import register_env
from PaintRLEnv.robot_gym_env import RobotGymEnv


class PaintModel(Model):

    def _build_layers(self, inputs, num_outputs, options):
        pass

    def _build_layers_v2(self, input_dict, num_outputs, options):
        scaled_images = tf.cast(input_dict['obs']['image'], tf.float32) / 255.
        conv1 = tf.layers.conv2d(inputs=scaled_images, filters=32, strides=(4, 4), kernel_size=(8, 8), padding='VALID',
                                 activation=tf.nn.relu, name='conv1')

        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, strides=(2, 2), kernel_size=(4, 4), padding='VALID',
                                 activation=tf.nn.relu, name='conv2')

        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, strides=(1, 1), kernel_size=(3, 3), padding='VALID',
                                 activation=tf.nn.relu, name='conv3')

        conv3 = flatten(conv3)
        fc1 = tf.layers.dense(conv3, 512, activation=tf.nn.relu, name='fc1')
        fc1 = tf.concat([fc1, input_dict['obs']['pose']], 1)
        fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name='fc2')
        fc3 = tf.layers.dense(fc2, 32, activation=tf.nn.relu, name='fc3')
        out = tf.layers.dense(fc3, 4, activation=tf.nn.tanh, name='out')
        return out, fc3


def env_creator(env_config):
    return RobotGymEnv(**env_config)


ModelCatalog.register_custom_model('paint_model', PaintModel)
register_env('robot_gym_env', env_creator)

ray.init(num_gpus=1)

agent = ppo.PPOAgent(env='robot_gym_env', config={
    'model': {
        'custom_model': 'paint_model',
        'custom_options': {},  # extra options to pass to your model
    },
    'env_config': {
                    'urdf_root': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PaintRLEnv'),
                    'renders': False,
                    'render_video': False,
                },
    'observation_filter': 'NoFilter',
    'vf_share_layers': True,
})


while True:
    print(agent.train())
