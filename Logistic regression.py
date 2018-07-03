import tensorflow as tf
import numpy as np


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

np.random.seed(0)       # random number generator

data = np.load("/Users/mihirumeshnimgade/Desktop/data_with_labels.npz")

train = data['arr_0'] / 255.         # holds pixel values scaled from 0 to 1
labels = data['arr_1']             # holds the type of font it was (0, 1, 2, 3, 4)


# print(train[1])
# print(labels[0])

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# plt.ion()
# plt.plot(train[0])
# plt.matshow(train[78])
# plt.ion()
# plt.show()

X = tf.placeholder(tf.float32, shape=(12, 784), name="X")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    n_inputs = tf.shape(X)[1]
    init.run()
    print(n_inputs.eval())