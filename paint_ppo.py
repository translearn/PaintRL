import numpy as np
import tensorflow as tf
import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import flatten
from PaintRLEnv.robot_gym_env import RobotGymEnv


class PaintModel(Model):

    def _build_layers(self, inputs, num_outputs, options):
        pass

    def _build_layers_v2(self, input_dict, num_outputs, options):
        conv1 = tf.layers.conv2d(inputs=input_dict['obs']['image'], filters=32, kernel_size=(5, 5), padding='SAME',
                                 activation=tf.nn.relu, name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2, name='pool1')

        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=(5, 5), padding='SAME',
                                 activation=tf.nn.relu, name='conv2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2, name='pool2')

        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=(5, 5), padding='SAME',
                                 activation=tf.nn.relu, name='conv3')
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 2), strides=2, name='pool3')
        pool3 = flatten(pool3)
        fc = tf.layers.dense(pool3, 64, activation=tf.nn.relu, name='fc')
        dropout = tf.layers.dropout(fc, tf.constant(0.75), training=input_dict["is_training"], name='dropout')
        out = tf.layers.dense(dropout, 2, name='logits')

        return out, fc


ModelCatalog.register_custom_model("paint_model", PaintModel)


ray.init(num_gpus=1)

agent = ppo.PPOAgent(env="CartPole-v0", config={
    "model": {
        "custom_model": "paint_model",
        "custom_options": {},  # extra options to pass to your model
    },
})



