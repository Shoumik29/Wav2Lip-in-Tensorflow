
Here We have Transformed the original [wav2lip](https://github.com/Rudrabha/Wav2Lip) model from pytorch to tensorflow

We also modified the make_video part for making videos from generated images

You can get the tensorflow weights and pytorch weights here in this google drive [link](https://drive.google.com/drive/folders/1OK2FUVBdd6y19i9E1jEQ6SQe42epgV8v)

Feel free to Contact with us for details here, Shomik20@gmail.com


Instructions:

1. install ffmpeg (video editor) "sudo apt-get install ffmpeg" (for linux machine)<br>
2. install requirements.txt (probably you dont need all of them)<br>
3. try to use videos which has faces in all the frames <br>
4. try to use videos and audios of 20 secs (max 30 sec)<br>
5. try to use videos longer than audios (must)<br>


You need all the files in this folder and also files from the [drive](https://drive.google.com/drive/folders/1OK2FUVBdd6y19i9E1jEQ6SQe42epgV8v) link to run the make_video.py program

The face detection part was from this [repo](https://github.com/1adrianb/face-alignment) which we found in the [wav2lip](https://github.com/Rudrabha/Wav2Lip) repo

