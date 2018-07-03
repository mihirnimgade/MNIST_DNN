import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)

c = a + b
d = a * b

vector1 = tf.constant([1.,2.])                      # 1D vector

matrix1 = tf.constant([[3., 7.], [2.,1.]])            # 2x2 matrix
matrix2 = tf.constant([[4.,5.], [9.,7.]])             # second 2x2 matrix

m1m2 = tf.matmul(matrix1, matrix2)                    # matrix multiplication


W = tf.Variable(0, name = "weight")                 # assigning variable value and name


sess = tf.InteractiveSession()

init_op = tf.global_variables_initializer()         # initialises all the variables

sess.run(init_op)                                   # runs the variable initializer


print("W is: ", end = "")
print(W.eval())                                     # evaluates W

print("m1m2 is: ", end = "")
print(m1m2.eval())                                  # evaluates the matrix multiplication

W += a                                              # increments W
print("W after adding a: ", end = "")
print(W.eval())

W += a
print("W after adding a: ", end = "")
print(W.eval())

E = d + b                                           # evaluates to 4

print("E is: ", end = "")
print(E.eval())

print("E and D are: ", end = "")
print(sess.run([E, d]))                             # shows values for E and d at the same time

print("E with custom d=4: ", end = "")
print(sess.run(E, feed_dict={d:4.}))                 # feeds in value for d


sess.close()                                        # closes the tf session



