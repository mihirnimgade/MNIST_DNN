import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import scipy.ndimage as scnd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import MinMaxScaler

mnist = input_data.read_data_sets("/tmp/data/")

batch_size = 1

init = tf.global_variables_initializer()
# #
with tf.Session() as sess:
    init.run()
# #     # saver.restore(sess, "./FirstDNN.ckpt")
    for iteration in range(mnist.train.num_examples // batch_size):
        X_batch, y_batch = mnist.train.next_batch(batch_size)
    print(X_batch)

