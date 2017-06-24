import tensorflow as tf
import numpy as np

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    cost_val, h_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={ x : x_data, y : y_data })
    print(step, "Cost : ", cost_val, "\nPrediction\n", h_val)

"""
0 Cost :  5.26825e+11 
Prediction
 [[  512999.25  ]
 [ 1031316.5625]
 [  811585.3125]
 [  569318.3125]
 [  670733.5625]
 [  676365.875 ]
 [  620014.    ]
 [  789032.9375]]
1 Cost :  5.78812e+26 
Prediction
 [[ -1.69707139e+13]
 [ -3.41637889e+13]
 [ -2.68754202e+13]
 [ -1.88395280e+13]
 [ -2.22033892e+13]
 [ -2.23902706e+13]
 [ -2.05214565e+13]
 [ -2.61278925e+13]]
2 Cost :  inf 
Prediction
 [[  5.62517186e+20]
 [  1.13240486e+21]
 [  8.90821999e+20]
 [  6.24461454e+20]
 [  7.35961222e+20]
 [  7.42155642e+20]
 [  6.80211373e+20]
 [  8.66044249e+20]]
3 Cost :  inf 
Prediction
 [[ -1.86453916e+28]
 [ -3.75350830e+28]
 [ -2.95274960e+28]
 [ -2.06986187e+28]
 [ -2.43944277e+28]
 [ -2.45997491e+28]
 [ -2.25465232e+28]
 [ -2.87062057e+28]]
4 Cost :  inf 
Prediction
 [[  6.18026690e+35]
 [  1.24415105e+36]
 [  9.78728785e+35]
 [  6.86083642e+35]
 [  8.08586226e+35]
 [  8.15391926e+35]
 [  7.47334934e+35]
 [  9.51505988e+35]]
5 Cost :  inf 
Prediction
 [[-inf]
 [-inf]
 [-inf]
 [-inf]
 [-inf]
 [-inf]
 [-inf]
 [-inf]]
6 Cost :  nan 
Prediction
 [[ nan]
 [ nan]
 [ nan]
 [ nan]
 [ nan]
 [ nan]
 [ nan]
 [ nan]]
"""