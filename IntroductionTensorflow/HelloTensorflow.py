import tensorflow as tf

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    x = 10
    y = 2
    z = tf.sub(tf.div(x, y), 1)
    output = sess.run(z)
    print(output)