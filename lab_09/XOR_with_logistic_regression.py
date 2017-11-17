import tensorflow as tf
import numpy as np

#  XOR data set
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0],    [1],    [1],    [0]], dtype=np.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid : tf.div(1., 1. +  tf.exp(tf.matmul(x, w)))
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost/loss function
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# launch graph
with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={ x : x_data, y : y_data })
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={ x : x_data, y : y_data }), sess.run(w))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={ x : x_data, y : y_data })
    print('\nHypothesis : ', h, '\nCorrect : ', c, '\nAccuracy : ', a)