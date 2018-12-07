import baselines.ppo2.ppo2 as ppo
import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc, conv_to_fc
from baselines.common.models import register
from PaintRLEnv.robot_gym_env import RobotGymEnv


hyper_params = dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy')


model = ppo.learn(
    network='conv_with_fc',
    env=RobotGymEnv,
    total_timesteps=1e4,
    **hyper_params
)


@register('conv_with_fc')
def conv_with_fc(convs=((32, 8, 4), (64, 4, 2), (64, 3, 1)), fc_layers=2, num_hidden=64, layer_norm=False, **conv_kwargs):
    """
    customized for the paint robot task, copied from baselines.common.models
    :param convs: list of triples (filter_number, filter_size, stride) specifying parameters for each layer.
    :param fc_layers: int, number of fully-connected layers (default: 2)
    :param num_hidden: int, size of fully-connected layers (default: 64)
    :param layer_norm:
    :param conv_kwargs:
    :return: function
    """

    def network_fn(img, pos):
        out = tf.cast(img, tf.float32) / 255.
        with tf.variable_scope("conv_fc_net"):
            for num_outputs, kernel_size, stride in convs:
                out = tf.contrib.layers.convolution2d(out, num_outputs=num_outputs, kernel_size=kernel_size,
                                                      stride=stride, activation_fn=tf.nn.relu, **conv_kwargs)

            # h = tf.contrib.layers.spatial_softmax(out)
            h = conv_to_fc(out)
            h = tf.concat(h, pos)
            for i in range(fc_layers):
                h = fc(h, 'conv_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
                if layer_norm:
                    h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
                h = tf.nn.relu(h)
            h = fc(h, 'conv_fc{}'.format(fc_layers + 1), nh=num_hidden, init_scale=np.sqrt(2))
            h = tf.nn.
        return h
    return network_fn

