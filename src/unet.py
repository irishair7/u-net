from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def unet_arg_scope(weight_decay=0.0005):
  """Defines the UNET arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME') as arg_sc:
      return arg_sc


def unet(inputs,
           is_training=True,
           spatial_squeeze=True,
           scope='unet'):
    """
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """

    end_points = {}

    with tf.variable_scope(scope, 'unet', [inputs]):
        with slim.arg_scope([slim.max_pool2d], stride=2, padding='SAME'):
            end_point = 'Conv2d_1a'
            net = slim.repeat(inputs, 2, slim.conv2d, 32, [3, 3], scope='conv1')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())
            end_point = 'Conv2d_1b'
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            end_point = 'Conv2d_2a'
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv2')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())
            end_point = 'Conv2d_2b'
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            end_point = 'Conv2d_3a'
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv3')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())
            end_point = 'Conv2d_3b'
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            end_point = 'Conv2d_4a'
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv4')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())
            end_point = 'Conv2d_4b'
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            end_point = 'Conv2d_5a'
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            shape = net.get_shape().as_list()
            up = tf.image.resize_images(net, [shape[1]*2, shape[2]*2])
            print('up', up.get_shape().as_list())

            end_point = 'merge_1'
            net = tf.concat(axis=3, values=[up, end_points['Conv2d_4a']])
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            end_point = 'Conv2d_6a'
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv6')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            shape = net.get_shape().as_list()
            up = tf.image.resize_images(net, [shape[1]*2, shape[2]*2])
            print('up', up.get_shape().as_list())

            end_point = 'merge_2'
            net = tf.concat(axis=3, values=[up, end_points['Conv2d_3a']])
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            end_point = 'Conv2d_7a'
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv7')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            shape = net.get_shape().as_list()
            up = tf.image.resize_images(net, [shape[1]*2, shape[2]*2])
            print('up', up.get_shape().as_list())

            end_point = 'merge_3'
            net = tf.concat(axis=3, values=[up, end_points['Conv2d_2a']])
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            end_point = 'Conv2d_8a'
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv8')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            shape = net.get_shape().as_list()
            up = tf.image.resize_images(net, [shape[1]*2, shape[2]*2])
            print('up', up.get_shape().as_list())

            end_point = 'merge_4'
            net = tf.concat(axis=3, values=[up, end_points['Conv2d_1a']])
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            end_point = 'Conv2d_9a'
            net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='conv9')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            end_point = 'Conv2d_10a'
            net = slim.conv2d(net, 1, [1, 1], activation_fn=tf.nn.sigmoid, padding='VALID', scope='conv10')
            end_points[end_point] = net
            print(end_point, net.get_shape().as_list())

            return net