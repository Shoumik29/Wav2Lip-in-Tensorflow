from Keras_layers import conv2d_block, conv2d_transpose
from tensorflow.keras.layers import Concatenate, Conv2D, Activation
import tensorflow as tf
from dataProcess import getitem
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2



def loading_face_dec_weights():

	torch.cuda.empty_cache()

	all_weights = []
	
	torch_model = torch.load('wav2lip.pth')
	j = 0
	for i in torch_model['state_dict']:
		if j % 7 == 0:
			w = torch_model['state_dict'][i].cpu().numpy()
			w = np.moveaxis(w, [0,1], [-1,-2])
			all_weights.append(w)
		
		if (j-1) % 7 == 0:
			w = torch_model['state_dict'][i].cpu().numpy()
			w =  np.transpose(w)
			all_weights.append(w)
		
		if (j-2) % 7 == 0:
			w = torch_model['state_dict'][i].cpu().numpy()
			w =  np.transpose(w)
			all_weights.append(w)
			
		if (j-3) % 7 == 0:
			w = torch_model['state_dict'][i].cpu().numpy()
			w =  np.transpose(w)
			all_weights.append(w)
			
		if (j-4) % 7 == 0:
			w = torch_model['state_dict'][i].cpu().numpy()
			w =  np.transpose(w)
			all_weights.append(w)
		
		if (j-5) % 7 == 0:
			w = torch_model['state_dict'][i].cpu().numpy()
			w =  np.transpose(w)
			all_weights.append(w)
				
		j += 1

	print(len(all_weights))
	
	all_weights.reverse()
	
	inp1 = tf.zeros([5, 96, 96, 6], tf.dtypes.float32)
	inp2 = tf.zeros([5, 80, 16, 1], tf.dtypes.float32) 

	model = encoderDecoder()
	
	out = model(inp2, inp1)
	
	i=0
	indx = [6,18,24,18,18,12,12,78,6,12,18,18,18,18,18,8]
	for k in indx:
		weight = []
		for j in range(k):
			weight.append(all_weights[-1])
			all_weights.pop()
		model.layers[i].set_weights(weight)
		i += 1
		
	
	print("weights are initialized")
	
	
	return model
	
	

class encoderDecoder(tf.keras.Model):
	def __init__(self):
		super(encoderDecoder, self).__init__()
		
		self.face_encoder_blocks = [
		
			tf.keras.Sequential([conv2d_block(16, (7,7), (1,1), 3),]), #96,96

			tf.keras.Sequential([conv2d_block(32, (3,3), (2,2), 1),
			conv2d_block(32, (3,3), (1,1), 1, residual=True),
			conv2d_block(32, (3,3), (1,1), 1, residual=True),]),
	
			tf.keras.Sequential([conv2d_block(64, (3,3), (2,2), 1),
			conv2d_block(64, (3,3), (1,1), 1, residual=True),
			conv2d_block(64, (3,3), (1,1), 1, residual=True),
			conv2d_block(64, (3,3), (1,1), 1, residual=True),]),
	
			tf.keras.Sequential([conv2d_block(128, (3,3), (2,2), 1),
			conv2d_block(128, (3,3), (1,1), 1, residual=True),	
			conv2d_block(128, (3,3), (1,1), 1, residual=True),]),
	
			tf.keras.Sequential([conv2d_block(256, (3,3), (2,2), 1),
			conv2d_block(256, (3,3), (1,1), 1, residual=True),
			conv2d_block(256, (3,3), (1,1), 1, residual=True),]),
		
			tf.keras.Sequential([conv2d_block(512, (3,3), (2,2), 1),
			conv2d_block(512, (3,3), (1,1), 1, residual=True),]),

			tf.keras.Sequential([conv2d_block(512, (3,3), (1,1), 0),
			conv2d_block(512, (1,1), (1,1), 0),])]
			
			
		self.audio_encoder = tf.keras.Sequential([
		
			conv2d_block(32, (3,3), (1,1), 1),
			conv2d_block(32, (3,3), (1,1), 1, residual=True),
			conv2d_block(32, (3,3), (1,1), 1, residual=True),

			conv2d_block(64, (3,3), (3,1), 1),
			conv2d_block(64, (3,3), (1,1), 1, residual=True),
			conv2d_block(64, (3,3), (1,1), 1, residual=True),
	
			conv2d_block(128, (3,3), (3,3), 1),
			conv2d_block(128, (3,3), (1,1), 1, residual=True),
			conv2d_block(128, (3,3), (1,1), 1, residual=True),
	
			conv2d_block(256, (3,3), (3,2), 1),
			conv2d_block(256, (3,3), (1,1), 1, residual=True),
	
			conv2d_block(512, (3,3), (1,1), 0),
	
			conv2d_block(512, (1,1), (1,1), 0)])
			
		
		self.face_decoder_blocks = [
			
			tf.keras.Sequential([conv2d_block(512, (1,1), (1,1), 0),]),
			
			tf.keras.Sequential([conv2d_transpose(512, (3,3), (1,1), 0),
			conv2d_block(512, (3,3), (1,1), 1, residual=True),]),
			
			tf.keras.Sequential([conv2d_transpose(512, (3,3), (2,2), 1, 1),	
			conv2d_block(512, (3,3), (1,1), 1, residual=True),
			conv2d_block(512, (3,3), (1,1), 1, residual=True),]),
			
			tf.keras.Sequential([conv2d_transpose(384, (3,3), (2,2), 1, 1),
			conv2d_block(384, (3,3), (1,1), 1, residual=True),
			conv2d_block(384, (3,3), (1,1), 1, residual=True),]),
			
			tf.keras.Sequential([conv2d_transpose(256, (3,3), (2,2), 1, 1),
			conv2d_block(256, (3,3), (1,1), 1, residual=True),
			conv2d_block(256, (3,3), (1,1), 1, residual=True),]),
			
			tf.keras.Sequential([conv2d_transpose(128, (3,3), (2,2), 1, 1),
			conv2d_block(128, (3,3), (1,1), 1, residual=True),
			conv2d_block(128, (3,3), (1,1), 1, residual=True),]),
			
			tf.keras.Sequential([conv2d_transpose(64, (3,3), (2,2), 1, 1),
			conv2d_block(64, (3,3), (1,1), 1, residual=True),
			conv2d_block(64, (3,3), (1,1), 1, residual=True),])]
			
		
		
		self.output_block = tf.keras.Sequential([
			conv2d_block(32, (3,3), (1,1), 1),
			Conv2D(3, kernel_size=(1,1), strides=(1,1), padding="valid"),
			Activation('sigmoid')
		])
	

	def call(self, audio_sequences, face_sequences):
			
		audio_embedding = self.audio_encoder(audio_sequences)
		
		feats = []
		x = face_sequences
        
		for f in self.face_encoder_blocks:
			x = f(x)
			feats.append(x)
        
		x = audio_embedding	
		
		for f in self.face_decoder_blocks:
			x = f(x)
			try:
				x = Concatenate(axis=3)([x, feats[-1]])
			except Exception as e:
				raise e
					
                
			feats.pop()
            
		x = self.output_block(x)
		
		return x
	
	
def main():

	model = loading_face_dec_weights()
		
	face_seq, audio_seq = getitem()
	generated_img_ten = model(audio_seq, face_seq)
	generated_img = generated_img_ten.cpu().numpy()
	model.summary()
	model.save('saved_model_acc/lip_Sync')
	print("Model saved successfully")

	
	
	plt.figure(figsize = (20, 20))
	for i in range(5):
		plt.subplot(3, 3, i + 1)
		plt.axis('off')
		img = cv2.cvtColor(generated_img[i], cv2.COLOR_BGR2RGB)
		plt.imshow(img)	
	plt.show()
	
	
	
if __name__ == '__main__':
	main()
	
