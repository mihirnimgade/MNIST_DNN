import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import scipy.ndimage as scnd

n_inputs = 28*28            # input neurons
n_hidden1 = 300             # hidden neurons in first hidden layer
n_hidden2 = 100             # hidden neurons in second hidden layer
n_outputs = 10             # output neurons
learning_rate = 0.01        # defines learning rate for gradient descent

mnist = input_data.read_data_sets("/tmp/data/")        # reads MNIST data
n_epochs = 100                                          # number of epochs to run the NN
batch_size = 50                                     # size of the batches of instances

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")    # placeholder with shape (undefined x 784), name "X", type 32 bit float
y = tf.placeholder(tf.int64, shape=(None), name="y")                # placeholder with no shape containing the labels for the data (i.e. what number it is if data is MNIST)

def neuronLayer(X, n_neurons, name, activation=None):                       # defines function to create a neural network layer
    with tf.name_scope(name):                                                   # creates name scope to make TensorFlow graph look nice
        n_inputs = int(X.shape[1])                                        # get number of inputs for the layer
        stddev = 2 / np.sqrt(n_inputs)                                           # defines a value for standard deviation
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)     # initialises random values for weights with size (n_inputs, n_neurons)
        W = tf.Variable(init, name="kernel")                                # variable 'W' contains weights for layer
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")                 # contains biases for layer initilised with zeroes
        Z = tf.matmul(X, W) + b                                        # matrix multiplication and addition of bias
        if activation is not None:          # if an activation function is given...
            return activation(Z)            # use it.
        else:                               # if not...
            return Z                        # return raw logit.

# with tf.name_scope("DNN"):
#     hidden1 = neuronLayer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
#     hidden2 = neuronLayer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
#     logits = neuronLayer(hidden2, n_outputs, name="outputs")

with tf.name_scope("DNN"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)              # builds first hidden layer
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)        # builds second hidden layer
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")                            # builds last output layer w/o softmax

with tf.name_scope("Loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)      # applies softmax activation f(x) and then computes the cross entropy
    loss = tf.reduce_mean(xentropy, name="Loss")                                            # finds the cross entropy for all instances
tf.summary.scalar('xentropy', loss)

with tf.name_scope("Train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)                    # implements the gradient descent optimizer to adjust weights; creates GradientDescentOptimizer object
    training_op = optimizer.minimize(loschs)                              #

with tf.name_scope("Evaluation"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
tf.summary.scalar('Accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("/Users/mihirumeshnimgade/Desktop/Project_1", graph=tf.get_default_graph())

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})

        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        summary = sess.run(merged, feed_dict={X: X_batch, y:y_batch})
        train_writer.add_summary(summary, epoch)

        acc_test = accuracy.eval(feed_dict={X:mnist.test.images, y:mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        save_path = saver.save(sess, "./FirstDNN.ckpt")

# number = scnd.imread("number7.png")
#
# flattened_number = number.reshape((1,784))
# flattened_number = flattened_number.astype(float)
#
#
# flattened_number = np.interp(flattened_number, (flattened_number.min(), flattened_number.max()), (1, 0))

# tf.reset_default_graph()

# init = tf.global_variables_initializer()
# saver = tf.train.Saver()

# with tf.Session() as sess:
#     init.run()
#     saver.restore(sess, "./FirstDNN.ckpt")
#     X_new_scaled = flattened_number
#     # Z = logits.eval(feed_dict={X:X_new_scaled})
#     Z = sess.run(logits, feed_dict={X:X_new_scaled})
#     y_pred = np.argmax(Z, axis=1)
#     print(y_pred)