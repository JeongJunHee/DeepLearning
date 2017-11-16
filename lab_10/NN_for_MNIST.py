import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_normal([784, 128]))
b1 = tf.Variable(tf.random_normal([128]))
l1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([128, 128]))
b2= tf.Variable(tf.random_normal([128]))
l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

w3 = tf.Variable(tf.random_normal([128, 10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(l2, w3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={ x : batch_xs, y : batch_ys })
            avg_cost += c / total_batch

        print('Epoch : ', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    print("Accuracy : ", sess.run(accuracy, feed_dict={ x : mnist.test.images, y : mnist.test.labels }))