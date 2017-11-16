## 5 steps of using TensorBoard

1. From TF graph, decide which tensors you want to log 

<pre><code>w_hist = tf.summary.histogram('weight', w)
cost_summ = tf.summary.scalar('cost', cost)</code></pre>  

2. Merge all summaries

<pre><code>summary = tf.summary.merge_all()</code></pre>

3. Create writer and add graph

<pre><code># Create summary writer
writer = tf.summary.FileWriter(./logs')
writer.add_graph(sess.graph)</code></pre>

4. Run summary merge and add_summary

<pre><code>s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
writer.add_summary(s, global_step=global_step)</code></pre>

5. Launch TensorBoard

<pre><code>tensorboard --logdir=./logs</code></pre>
