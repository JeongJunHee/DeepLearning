import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholder
x = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([8, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# hypothesis using sigmoid
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost/loss function
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * (tf.log(1 - hypothesis)))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = { x : x_data, y : y_data }
    for step in range(10001):
        sess.run(train, feed_dict=feed)
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict=feed))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={ x : x_data, y : y_data})
    print("\nHypothesis : ", h, "\nCorrect (y) : ", c, "\nAccuracy : ", a)