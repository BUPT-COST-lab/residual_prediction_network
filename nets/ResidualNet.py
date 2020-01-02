import tensorflow as tf
from layers.ResBlock import ResBlock as RB
import pdb
import numpy as np

def ResNet(images,keep_prob, seq_length, input_length, stacklength, num_hidden,filter_size):
	with tf.variable_scope('TrajNet', reuse=False):
		print 'ResNet'
		#print 'is_training', is_training
		h = images[:,0:seq_length,:,:]
		gt_images=images[:,seq_length:]
		dims=gt_images.shape[1]*gt_images.shape[2]*gt_images.shape[3]
		inputs = h
		if keep_prob is not 1:
			training = True
		else:
			training = False
		inputs = tf.layers.conv2d(inputs,num_hidden,filter_size, padding='same',activation=tf.nn.leaky_relu,  
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='h0')
		for i in range(stacklength):
			inputs=RB('TrajBlock'+str(i),filter_size,num_hidden,keep_prob)(inputs)
		fc_in=tf.layers.flatten(inputs)
		fc_in = tf.layers.dense(inputs=fc_in, units=300, activation=tf.nn.leaky_relu)
		fc_in = tf.layers.dense(inputs=fc_in, units=dims, activation=None)		
		out = tf.reshape(fc_in,gt_images.shape)	
		gen_images = out
		loss = tf.reduce_mean(tf.norm(gen_images-gt_images, axis=3, keep_dims=True, name='normal'))   # MPJPE loss

		return [gen_images, loss]

