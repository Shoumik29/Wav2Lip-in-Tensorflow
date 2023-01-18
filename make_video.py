from tensorflow import keras
import tensorflow as tf
import torch, face_detection
from tqdm import tqdm
import cv2
from glob import glob
from os.path import join, basename
import numpy as np
import subprocess
import platform
import audio
import random




mel_step_size = 16


video_source = 'input_data/cut-3.mp4'

audio_src = 'input_data/audio.wav'

face_detection_device = 'cpu'

inference_batch = 1	
	




def datagen(frames):
	
	coord, faces = face_detect(frames)
	face_data = []
	for i in range(0,len(faces)):
		img = faces[i].copy()
		img_resized = cv2.resize(img, (96, 96))
		img_mask = img_resized
		img_mask[img_mask.shape[0]//2:, :, :] = 0.
		
		wrong_img = random.choice(faces)
		
		while np.array_equal(wrong_img, faces[i]):
			wrong_img = random.choice(faces)
			
		wrong_img = cv2.resize(wrong_img, (96, 96))
		img = np.concatenate((img_mask, wrong_img), axis=2) / 255.
		face_data.append(img)
	
	return face_data, coord



def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def get_frame_id(frame):
		return int(basename(frame).split('.')[0])


def loading_gen_img():
	img_names = list(glob(join(img_path, '*.jpg')))
	
	img_dirs = []
    
	for i in range(0,len(img_names)):
		for j in img_names:
			if i == get_frame_id(j):
				image = cv2.imread(j)
				img_dirs.append(image)
				break		
	return img_dirs

def process_video_file():
	video_stream = cv2.VideoCapture(video_source)
	FPS = video_stream.get(cv2.CAP_PROP_FPS)
	
	frames = []
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		
		frames.append(frame)
	
	return frames, FPS


def face_detect(video_frames):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=face_detection_device)	
	
	results = []
	
	v_frames = video_frames.copy()
	
	
	pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
	print("Detecting face co-ordinates.....")
	for i in tqdm(range(0, len(v_frames))): 
		preds = detector.get_detection_for_image(v_frames[i])
		y1 = max(0, preds[1] - pady1)
		y2 = min(v_frames[i].shape[0], preds[3] + pady2)
		x1 = max(0, preds[0] - padx1)
		x2 = min(v_frames[i].shape[1], preds[2] + padx2)
		results.append([x1, y1, x2, y2])
	
	boxes = np.array(results)
	boxes = get_smoothened_boxes(boxes, T=5)
	
	images = []
	for l in range(0, len(v_frames)):
		x1, y1, x2, y2 = boxes[l]
		images.append(v_frames[l][y1: y2, x1:x2])
	res = []
	for (x1, y1, x2, y2) in boxes:
		res.append((y1, y2, x1, x2))
		
		
	del detector
	torch.cuda.empty_cache()
	return res, images
	
	
	
	
	
def makeVideo():
	
	#getting the real frames
	frames, fps = process_video_file()
	
	#getting the audio
	wav = audio.load_wav(audio_src, 16000)
	mel = audio.melspectrogram(wav)
	
	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1
		
	
	
	mels = mel_chunks[:(len(mel_chunks))]
	v_frames = frames[:len(mels)]
	
	
	faces, res = datagen(v_frames)
	
	#loading Model
	model = keras.models.load_model('saved_model_acc/lip_Sync')
	
	
	frame_h, frame_w = v_frames[0].shape[:-1]
	v_out = cv2.VideoWriter('output_video/gen_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
	
	j=0
	print("Putting gen faces on real faces......")
	while tqdm(j < len(faces)):
	
		inp_face_list = []
		inp_mel_list = []
		for k in range(0,inference_batch):
			inp_face_list.append(faces[j])
			inp_mel_list.append(mels[j])
			j = j + 1
			
		j = j - inference_batch
		
		face_inp = np.asarray(inp_face_list)
		audio_inp = np.asarray(inp_mel_list)
		
		face_inp = tf.convert_to_tensor(face_inp, dtype='float32')
		audio_inp = tf.convert_to_tensor(audio_inp, dtype='float32')
		audio_inp = tf.expand_dims(audio_inp, axis=3)
		
		
		gen_faces = model(audio_inp, face_inp)
		
		gen_faces = gen_faces.cpu().numpy() * 255.
		
		for w in range(0,inference_batch):
			y1, y2, x1, x2 = res[j]
			
			generated_face = cv2.resize(gen_faces[w], (x2 - x1, y2 - y1))
			v_frames[j][y1:y2, x1:x2] = generated_face
			v_out.write(v_frames[j])			
			j = j + 1
	
	v_out.release()							
											
	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_src, 'output_video/gen_video.avi', 'output_video/gen_video.mp4')
	subprocess.call(command, shell=platform.system() != 'Windows')
											
makeVideo()
#fg()

