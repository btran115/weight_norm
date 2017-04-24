from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np 
import scipy
import cPickle as pickle

from tensorflow.examples.tutorials.mnist import input_data

#one convolutional layer followed by output

train_size = 55000
test_size = 10000
valid_size = 5000
batch_size = 100
im_size = 28 #28*28
in_chan = 1
num_labels = 10

#mnist input, 1 channel, 28x28 images
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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
	dataset = tf.placeholder(tf.float32, 
								   shape = (None, im_size,im_size, in_chan))
	labels = tf.placeholder(tf.float32, shape = (None, num_labels))

	#add Gaussian noise
	g_noise = tf.truncated_normal((1, im_size,im_size, in_chan), stddev = 0.15)
	cur_layer = dataset + g_noise

	#g1 = tf.Variable(tf.truncated_normal([1, 1, in_chan, 96]))
	v1 = tf.Variable(tf.truncated_normal([3, 3, in_chan, 96], stddev = 0.1))
	b1 = tf.Variable(tf.truncated_normal([96]))
	cur_layer = conv(cur_layer, v1)
	cur_layer = maxpool(lrelu(cur_layer + b1))

	#g2 = tf.Variable(tf.truncated_normal([1, 1, 96, 96], stddev = 0.05))
	v2 = tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev = 0.01))
	b2 = tf.Variable(tf.truncated_normal([96]))
	cur_layer = conv(cur_layer, v2)
	cur_layer = maxpool(lrelu(cur_layer + b2))
	'''
	#g3 = tf.Variable(tf.truncated_normal([1, 1, 96, 96], stddev = 0.05))
	v3 = tf.Variable(tf.truncated_normal([3, 3, 96, 96], stddev = 0.05))
	b3 = tf.Variable(tf.truncated_normal([96]))
	cur_layer = conv(cur_layer, v3)
	cur_layer = lrelu(cur_layer + b3)
	cur_layer = tf.nn.dropout(maxpool(cur_layer), keep_prob = 0.5)

	#g4 = tf.Variable(tf.truncated_normal([1, 1, 96, 192]))
	v4 = tf.Variable(tf.truncated_normal([3, 3, 96, 192], stddev = 0.05))
	b4 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv(cur_layer, v4)
	cur_layer = lrelu(cur_layer + b4)

	#g5 = tf.Variable(tf.truncated_normal([1, 1, 192, 192]))
	v5 = tf.Variable(tf.truncated_normal([3, 3, 192, 192], stddev = 0.05))
	b5 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv(cur_layer, v5)
	cur_layer = lrelu(cur_layer + b5)

	#g6 = tf.Variable(tf.truncated_normal([1, 1, 192, 192]))
	v6 = tf.Variable(tf.truncated_normal([3, 3, 192, 192], stddev = 0.05))
	b6 = tf.Variable(tf.truncated_normal([192]))
	cur_layer = conv(cur_layer, v6)
	cur_layer = lrelu(cur_layer + b6)
	'''
	#cur_layer = tf.nn.avg_pool(cur_layer, 
	#						   [1, cur_layer.shape[1], cur_layer.shape[2], 1],
	#						   [1,1,1,1],
	#						   'VALID')
	cur_layer = tf.reshape(cur_layer, [-1, 7*7*96])

	vfc = tf.Variable(tf.truncated_normal([7*7*96,1024], stddev = 0.01))
	bfc = tf.Variable(tf.truncated_normal([1024]))
	cur_layer = lrelu(tf.matmul(cur_layer, vfc) + bfc)

	#g7 = tf.Variable(tf.truncated_normal([1,10]))
	v7 = tf.Variable(tf.truncated_normal([1024,10], stddev = 0.01))
	b7 = tf.Variable(tf.truncated_normal([10]))
	logits = tf.matmul(cur_layer, v7) + b7
	
	#loss and optimizer
	loss_all = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
	loss = tf.reduce_mean(loss_all)
	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

	grad = tf.gradients(loss, v7)

	#predictions for valid and test sets, do later

num_steps = 1501
all_grads = {}
all_wts = {}
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (mnist.train.labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_unflat[offset:(offset + batch_size)]
    batch_labels = mnist.train.labels[offset:(offset + batch_size)]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {dataset : batch_data, labels : batch_labels}
    _, l = session.run(
      		 [optimizer, loss], feed_dict=feed_dict)
    if step % 500 == 0:
      print("Minibatch loss at step %d: %f" % (step, l))
      #print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      #print("Validation accuracy: %.1f%%" % accuracy(
        #valid_prediction.eval(), valid_labels))
    if step == 300 or step ==700:
    	cur_grads = []
    	cur_wts = []
    	for i in range(50):
	    	g7,w7, = session.run([grad,v7], 
	    		feed_dict = {
	    		dataset : batch_data[i:i+1], 
	    		labels : batch_labels[i:i+1]})
    		cur_grads.append(g7[0])
    		cur_wts.append(w7)
		all_grads[step] = np.array(cur_grads)
		all_wts[step] = np.array(cur_wts)
		with open('grad7.pickle', 'w') as f:
			pickle.dump(all_grads, f)
		with open('wts7.pickle', 'w') as f:
			pickle.dump(all_wts, f)
 # print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))