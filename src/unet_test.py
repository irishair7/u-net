from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os.path
import unet
import os
import cv2

slim = tf.contrib.slim

pretrained_ckpt_path = '/home/martin/projects/sl_dl/ckpt/'

def predict(image):
    with tf.Graph().as_default():
        x = tf.placeholder('float')
        cwd = os.getcwd()
        print(image.shape)
        r = image[20:580, :, 0]
        g = image[20:580, :, 1]
        b = image[20:580, :, 2]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        er = clahe.apply(r).astype(np.uint8)
        eg = clahe.apply(g).astype(np.uint8)
        eb = clahe.apply(b).astype(np.uint8)
        img = np.stack((er, eg, eb), axis=2)
        mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

        with slim.arg_scope(unet.unet_arg_scope()):
            #get predict
            x = tf.reshape(x, shape=[-1, 560, 800, 3])
            pred, endpoints = unet.unet(x)

        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, "ckpt/model.ckpt-36000")

            batch_x = img
            batch_x = np.reshape(batch_x, (1, 560, 800, 3)).astype(np.float32)
            result = sess.run(pred, feed_dict={x:batch_x})

            print(result.shape)
            mask = np.reshape(result, (result.shape[1], result.shape[2]))
            sess.close()

        return mask.astype(np.uint8)
