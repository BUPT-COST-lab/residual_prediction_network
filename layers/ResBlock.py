import tensorflow as tf

class ResBlock():
	def __init__(self,layer_name, filter_size,num_hidden,keep_prob):
		self.layer_name=layer_name
		self.filter_size=filter_size
		self.keep_prob = keep_prob
		self.num_hidden = num_hidden

	def __call__(self, h, reuse=False):
		with tf.variable_scope(self.layer_name, reuse=False):
			out=[]
			filter_size=self.filter_size
			if self.keep_prob is not 1:
				training=True
			else:
				training=False

			h1 = tf.layers.conv2d(h,self.num_hidden,filter_size, padding='same',activation=None,  
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='h0')
			h1 = tf.layers.batch_normalization(h1,training=training)
			h1 = tf.nn.leaky_relu(h1)
			h2 = tf.layers.conv2d(h1,self.num_hidden,filter_size, padding='same',activation=None,  
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='h1')
			if h.shape[-1] is not h2.shape[-1]:
				h = tf.layers.conv2d(h,h2.shape[-1],1, padding='same',activation=None,
						kernel_initializer=tf.contrib.layers.xavier_initializer(),
						name='h3')
			h2 = tf.add(h,h2)
			h2 = tf.layers.batch_normalization(h2,training=training)
			out = tf.nn.leaky_relu(h2)	
			return out

