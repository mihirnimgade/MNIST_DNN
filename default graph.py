import tensorflow as tf

x = tf.get_default_graph()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    sess.run(x)
    print(x)