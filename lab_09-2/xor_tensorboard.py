import tensorflow as tf
import numpy as np

#  XOR data set
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0],    [1],    [1],    [0]], dtype=np.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

with tf.name_scope('layer1') as scope:
    w1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
    b1 = tf.Variable(tf.random_normal([10]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

    w1_hist = tf.summary.histogram('weight1', w1)
    b1_hist = tf.summary.histogram('bias1', b1)
    layer1_hist = tf.summary.histogram('layer1', layer1);

with tf.name_scope('layer2') as scope:
    w2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
    b2 = tf.Variable(tf.random_normal([10]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

    w2_hist = tf.summary.histogram('weight2', w2)
    b2_hist = tf.summary.histogram('bias2', b2)
    layer2_hist = tf.summary.histogram('layer2', layer2);

with tf.name_scope('layer3') as scope:
    w3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
    b3 = tf.Variable(tf.random_normal([10]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, w3) + b3)

    w3_hist = tf.summary.histogram('weight3', w3)
    b3_hist = tf.summary.histogram('bias3', b3)
    layer3_hist = tf.summary.histogram('layer3', layer3);

with tf.name_scope('layer4') as scope:
    w4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
    b4 = tf.Variable(tf.random_normal([1]), name='bias4')
    hypothesis = tf.sigmoid(tf.matmul(layer3, w4) + b4)

    w4_hist = tf.summary.histogram('weight4', w4)
    b4_hist = tf.summary.histogram('bias4', b4)
    hypothesis_hist = tf.summary.histogram('hypothesis', hypothesis);

# cost/loss function
with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
    cost_summ = tf.summary.scalar('cost', cost)

with tf.name_scope('train') as scope:
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar('accuracy', accuracy)

# launch graph
with tf.Session() as sess:
    # tensorboard -- logdir = ./logs/Deep_NN_for_XOR
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/Deep_NN_for_XOR')
    writer.add_graph(sess.graph) # show the graph

    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        summary, _ = sess.run([merged_summary, train], feed_dict={ x : x_data, y : y_data })
        writer.add_summary(summary, global_step=step)
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={ x : x_data, y : y_data }), sess.run([w1, w2, w3, w4]))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={ x : x_data, y : y_data })
    print('\nHypothesis : ', h, '\nCorrect : ', c, '\nAccuracy : ', a)