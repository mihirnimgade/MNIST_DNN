import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import scipy.ndimage as scnd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import MinMaxScaler

number = scnd.imread("number7.png")

flattened_number = number.reshape((1,784))
flattened_number = flattened_number.astype(float)


flattened_number = np.interp(flattened_number, (flattened_number.min(), flattened_number.max()), (1, 0))

tf.reset_default_graph()

graph = tf.get_default_graph()
logits = graph.get_tensor_by_name("DNN/outputs/kernel:0")

init = tf.global_variables_initializer()
saver = tf.train.import_meta_graph("FirstDNN.ckpt.meta")



with tf.Session() as sess:
    init.run()
    saver.restore(sess, "./FirstDNN.ckpt")
    X_new_scaled = flattened_number
    Z = logits.eval(feed_dict={X:X_new_scaled})
    # Z = sess.run(logits, feed_dict={X:X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
