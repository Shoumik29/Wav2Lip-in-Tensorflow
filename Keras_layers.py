import tensorflow as tf
from tensorflow.keras import layers





class conv2d_block(tf.keras.layers.Layer):
	def __init__(self, filters, kernelSize, stride, pad, residual = False, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		self.y = tf.keras.Sequential([
			layers.ZeroPadding2D(padding=pad),
			layers.Conv2D(filters, kernel_size = kernelSize, strides = stride, padding = "valid"),
			layers.BatchNormalization()
			])
		
		self.act = layers.ReLU()
		self.residual = residual
	
	
	def call(self, x):
		out = self.y(x)
		if self.residual:
			out += x
		
		return self.act(out)
	
		
		

class conv2d_transpose(tf.keras.layers.Layer):
	def __init__(self, filters, kernelSize, stride, pad, outpad = None, Name = None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.y = tf.keras.Sequential([
			layers.Conv2DTranspose(filters, kernel_size = kernelSize, strides = stride, padding = 'valid', output_padding = outpad, name = Name),
			layers.BatchNormalization(),
			layers.Cropping2D(cropping=pad)
			])
		
		self.act = layers.ReLU()
	
	
	def call(self, x):
		out = self.y(x)
		return self.act(out)
		
		
		
		
		
		
		
		
