from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np 
import scipy

from tensorflow.examples.tutorials.mnist import input_data

train_size = 55000
test_size = 10000
valid_size = 5000
batch_size = 100
im_size = 28 #28*28
in_chan = 1

#mnist input, 1 channel, 28x28 images
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#change images to mean 0
for image_set in mnist:
	for im in image_set.images:
		im = im - np.mean(im)

'''
if we're using Cifar, we'll need to ZCA whiten the input
'''

#leaky relu. maybe add a cut off function if needed
def lrelu(x, alpha = 0.):
	negative_part = tf.nn.relu(-x)
	x = tf.nn.relu(x) + alpha*negative_part
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
	train_dataset = tf.constant(mnist.train.images)
	train_labels = tf.constant(mnist.train.labels)
	valid_dataset = tf.constant(mnist.validation.images)
	test_dataset = tf.constant(mnist.test.images)

	train_dataset = tf.reshape(train_dataset, [-1, im_size, im_size, in_chan])
	#add Gaussian noise
	g_noise = tf.truncated_normal(train_dataset.shape, stddev = 0.15)
	train_dataset = train_dataset + g_noise

	#first set of 3 3x3conv leaky relu, a 2x2 max pool w/ stride 2, dropout w/ p=0.5
	#TODO: figure out how to intialize
	cur_layer = train_dataset
	g1 = tf.Variable(tf.truncated_normal([1, 1, in_chan, 96]))
	v1 = tf.Variable(tf.truncated_normal([3, 3, in_chan, 96]))
	b1 = tf.Variable(tf.truncated_normal([96]))
	cur_layer = conv(cur_layer, tf.multiply(g1,v1/tf.norm(v1, axis = (0,1))))
	cur_layer = lrelu(cur_layer + b1)

	g2 = tf.Variable(tf.truncated_normal([1, 1, 96, 96]))
	v2 = tf.Variable(tf.truncated_normal([3, 3, 96, 96]))
	b2 = tf.Variable(tf.truncated_normal([96]))
	cur_layer = conv(cur_layer, tf.multiply(g2,v2/tf.norm(v2, axis = (0,1))))
	cur_layer = lrelu(cur_layer + b2)

	g3 = tf.Variable(tf.truncated_normal([1, 1, 96, 96]))
	v3 = tf.Variable(tf.truncated_normal([3, 3, 96, 96]))
	b3 = tf.Variable(tf.truncated_normal([96]))
	cur_layer = conv(cur_layer, tf.multiply(g3,v3/tf.norm(v3, axis = (0,1))))
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
	g7 = tf.Variable(tf.truncated_normal([1, 1, 96, 192]))
	v7 = tf.Variable(tf.truncated_normal([3, 3, 96, 192]))
	b7 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv_no_pad(cur_layer, tf.multiply(g7,v7/tf.norm(v7, axis = (0,1))))
	cur_layer = lrelu(cur_layer + b7)

	g8 = tf.Variable(tf.truncated_normal([1, 1, 192, 192]))
	v8 = tf.Variable(tf.truncated_normal([1, 1, 192, 192]))
	b8 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv(cur_layer, tf.multiply(g8,v8/tf.norm(v8, axis = (0,1))))
	cur_layer = lrelu(cur_layer + b8)

	g9 = tf.Variable(tf.truncated_normal([1, 1, 192, 192]))
	v9 = tf.Variable(tf.truncated_normal([1, 1, 192, 192]))
	b9 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv(cur_layer, tf.multiply(g9,v9/tf.norm(v9, axis = (0,1))))
	cur_layer = lrelu(cur_layer + b9)
	cur_layer = tf.nn.avg_pool(cur_layer, 
							   [1, cur_layer.shape[1], cur_layer.shape[2], 192],
							   [1,1,1,1],
							   'VALID')
	cur_layer = tf.reshape(cur_layer, [-1, 192])
	g10 = tf.Variable(tf.truncated_normal([1,10]))
	v10 = tf.Variable(tf.truncated_normal([192,10]))
	b10 = tf.Variable(tf.truncated_normal([10]))
	last_layer = tf.matmul(cur_layer, tf.multiply(g10, v10/tf.norm(v10, axis = 0))) + b10
	
