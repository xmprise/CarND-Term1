import datetime as dt
import input_data
import tensorflow as tf

# Get timestamp
timestamp = dt.datetime.now().strftime('%Y%m%d%H%M%S')
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 8000
batch_size = 100
display_step = 10

# X, Y
X = tf.placeholder(tf.float32, [None, 784], name = 'X-input')
Y = tf.placeholder(tf.float32, [None, 10], name = 'Y-input')

# Weights
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))      # 3x3x1 conv, 32 outputs
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))     # 3x3x32 conv, 64 outputs
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))    # 3x3x32 conv, 128 outputs
W4 = tf.Variable(tf.random_normal([2048, 625], stddev=0.01))        # FC 128 * 4 * 4 inputs, 625 outputs
W5 = tf.Variable(tf.random_normal([625, 10], stddev=0.01))          # FC 625 inputs, 10 outputs (labels)

tf.histogram_summary('Weight1', W1)
tf.histogram_summary('Weight2', W2)
tf.histogram_summary('Weight3', W3)
tf.histogram_summary('Weight4', W4)
tf.histogram_summary('Weight5', W5)

# Construct model
dropout_cnn_rate = tf.placeholder(tf.float32)
dropout_fcc_rate = tf.placeholder(tf.float32)
X_image = tf.reshape(X, [-1, 28, 28, 1], name='X-input-reshape')

with tf.name_scope('Layer1'):
    l1a = tf.nn.relu(tf.nn.conv2d(X_image, W1, strides=[1, 1, 1, 1], padding='SAME'))
    print(l1a)  # l1a shape=(?, 28, 28, 32)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(l1)  # l1 shape=(?, 14, 14, 32)
    l1 = tf.nn.dropout(l1, dropout_cnn_rate)

with tf.name_scope('Layer2'):
    l2a = tf.nn.relu(tf.nn.conv2d(l1, W2, strides=[1, 1, 1, 1], padding='SAME'))
    print(l2a)  # l2a shape=(?, 14, 14, 64)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(l2)  # l2 shape=(?, 7, 7, 64)
    l2 = tf.nn.dropout(l2, dropout_cnn_rate)

with tf.name_scope('Layer3'):
    l3a = tf.nn.relu(tf.nn.conv2d(l2, W3, strides=[1, 1, 1, 1], padding='SAME'))
    print(l3a)  # l3a shape=(?, 7, 7, 128)
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(l3)  # l3 shape=(?, 4, 4, 128)
    l3 = tf.reshape(l3, [-1, W4.get_shape().as_list()[0]])
    print(l3)  # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, dropout_cnn_rate)

with tf.name_scope('Layer4'):
    l4 = tf.nn.relu(tf.matmul(l3, W4))
    l4 = tf.nn.dropout(l4, dropout_fcc_rate)

with tf.name_scope('Layer5'):
    hypothesis = tf.matmul(l4, W5)

# Minimize error using cross entropy
with tf.name_scope('Cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
    tf.scalar_summary('Cost', cost)

# Gradient Descent
with tf.name_scope('Train'):
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

# Accuracy
with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32))
    tf.scalar_summary('Accuracy', accuracy)

# Launch the graph
with tf.Session() as sess:
    writer = tf.train.SummaryWriter('./logs/cnn_' + timestamp, sess.graph)
    merged = tf.merge_all_summaries()

    tf.initialize_all_variables().run()

    # Training cycle
    for epoch in range(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # Fit training using batch data
        _, summary = sess.run([optimizer, merged],
                              feed_dict={X: batch_xs, Y: batch_ys, dropout_cnn_rate: 0.7, dropout_fcc_rate: 0.5})

        # Summarize
        writer.add_summary(summary, epoch)

    writer.close()
    print('Accuracy', accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_cnn_rate: 1,
                                     dropout_fcc_rate: 1}))