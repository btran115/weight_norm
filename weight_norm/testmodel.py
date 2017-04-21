from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np 
import scipy

from tensorflow.examples.tutorials.mnist import input_data

#full model

train_size = 55000
test_size = 10000
valid_size = 5000
batch_size = 100
im_size = 28 #28*28
in_chan = 1
num_labels = 10

#mnist input, 1 channel, 28x28 images
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#change images to mean 0, reshape to 28x28
for im in mnist.train.images:
	im = im - np.mean(im)
for im in mnist.validation.images:
	im = im - np.mean(im)
for im in mnist.test.images:
	im = im - np.mean(im)
train_unflat = np.reshape(mnist.train.images, (train_size, im_size, im_size, in_chan))
valid_unflat = np.reshape(mnist.validation.images, 
									 (valid_size, im_size, im_size, in_chan))
test_unflat = np.reshape(mnist.test.images, (test_size, im_size, im_size, in_chan))
'''
if we're using Cifar, we'll need to ZCA whiten the input
'''

#leaky relu. maybe add a cut off function if needed
def lrelu(x, alpha = 0.):
	negative_part = tf.nn.relu(-x)
	x = tf.nn.relu(x) - alpha*negative_part
	return x

def conv(x,W):
	return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding = 'SAME')

def conv_no_pad(x,W):
	return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding = 'VALID')

def maxpool(x):
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

#build our model
graph = tf.Graph()
with graph.as_default():
	train_dataset = tf.placeholder(tf.float32, 
								   shape = (batch_size, im_size,im_size, in_chan))
	train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
	valid_dataset = tf.constant(valid_unflat)
	test_dataset = tf.constant(test_unflat)

	#add Gaussian noise
	g_noise = tf.truncated_normal(train_dataset.shape, stddev = 0.15)
	train_dataset = train_dataset + g_noise

	#first set of 3 3x3conv leaky relu, a 2x2 max pool w/ stride 2, dropout w/ p=0.5
	#TODO: figure out how to intialize
	cur_layer = train_dataset
	#g1 = tf.Variable(tf.truncated_normal([1, 1, in_chan, 96], stddev = 0.05))
	v1 = tf.Variable(tf.truncated_normal([3, 3, in_chan, 96], stddev = 0.05))
	b1 = tf.Variable(tf.truncated_normal([96]))
	cur_layer = conv(cur_layer, v1)
	cur_layer = lrelu(cur_layer + b1)

	#g2 = tf.Variable(tf.truncated_normal([1, 1, 96, 96], stddev = 0.05))
	v2 = tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev = 0.05))
	b2 = tf.Variable(tf.truncated_normal([96]))
	cur_layer = conv(cur_layer, v2)
	cur_layer = lrelu(cur_layer + b2)

	#g3 = tf.Variable(tf.truncated_normal([1, 1, 96, 96], stddev = 0.05))
	v3 = tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev = 0.05))
	b3 = tf.Variable(tf.truncated_normal([96]))
	cur_layer = conv(cur_layer, v3)
	cur_layer = lrelu(cur_layer + b3)
	cur_layer = tf.nn.dropout(maxpool(cur_layer), keep_prob = 0.5)
 	''' don't have the power to run this yet
	#second set of the same
	g4 = tf.Variable(tf.truncated_normal([1, 1, 96, 192]))
	v4 = tf.Variable(tf.truncated_normal([3, 3, 96, 192]))
	b4 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv(cur_layer, tf.multiply(g4,v4/tf.norm(v4, axis = (0,1))))
	cur_layer = lrelu(cur_layer + b4)

	g5 = tf.Variable(tf.truncated_normal([1, 1, 192, 192]))
	v5 = tf.Variable(tf.truncated_normal([3, 3, 192, 192]))
	b5 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv(cur_layer, tf.multiply(g5,v5/tf.norm(v5, axis = (0,1))))
	cur_layer = lrelu(cur_layer + b5)

	g6 = tf.Variable(tf.truncated_normal([1, 1, 192, 192]))
	v6 = tf.Variable(tf.truncated_normal([3, 3, 192, 192]))
	b6 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv(cur_layer, tf.multiply(g6,v6/tf.norm(v6, axis = (0,1))))
	cur_layer = lrelu(cur_layer + b6)
	cur_layer = tf.nn.dropout(maxpool(cur_layer), keep_prob = 0.5)
	'''

	#3x3 conv, 2 sets of 1x1 conv, global average pool, softmax output
	#don't forget to change 96 to 192 when previous set is added back
	#g7 = tf.Variable(tf.truncated_normal([1, 1, 96, 192], stddev = 0.05))
	v7 = tf.Variable(tf.truncated_normal([3, 3, 96, 192], stddev = 0.05))
	b7 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv_no_pad(cur_layer, v7)
	cur_layer = lrelu(cur_layer + b7)

	#g8 = tf.Variable(tf.truncated_normal([1, 1, 192, 192], stddev = 0.05))
	v8 = tf.Variable(tf.truncated_normal([1, 1, 192, 192], stddev = 0.05))
	b8 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv(cur_layer, v8)
	cur_layer = lrelu(cur_layer + b8)

	#g9 = tf.Variable(tf.truncated_normal([1, 1, 192, 192], stddev = 0.05))
	v9 = tf.Variable(tf.truncated_normal([1, 1, 192, 192], stddev = 0.05))
	b9 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv(cur_layer, v9)
	cur_layer = lrelu(cur_layer + b9)
	cur_layer = tf.nn.avg_pool(cur_layer, 
							   [1, cur_layer.shape[1], cur_layer.shape[2], 1],
							   [1,1,1,1],
							   'VALID')
	cur_layer = tf.reshape(cur_layer, [-1, 192])
	g10 = tf.Variable(tf.truncated_normal([1,10], stddev = 0.05))
	v10 = tf.Variable(tf.truncated_normal([192,10], stddev = 0.05))
	b10 = tf.Variable(tf.truncated_normal([10]))
	logits = tf.matmul(cur_layer, tf.multiply(g10, v10/tf.norm(v10, axis = 0))) + b10
	
	#loss and optimizer
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels = train_labels, logits = logits))
	optimizer = tf.train.GradientDescentOptimizer(0.08).minimize(loss)

	#predictions for valid and test sets, do later

num_steps = 10000
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (mnist.train.labels.shape[0] - batch_size)
    offset = 0
    # Generate a minibatch.
    batch_data = train_unflat[offset:(offset + batch_size)]
    batch_labels = mnist.train.labels[offset:(offset + batch_size)]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run(
      [optimizer, loss], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      #print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      #print("Validation accuracy: %.1f%%" % accuracy(
        #valid_prediction.eval(), valid_labels))
 # print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))