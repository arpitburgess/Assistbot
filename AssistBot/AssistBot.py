import openai
from gtts import gTTS
import os
import pygame
import time

# defining voice input for Assistbot






# defining the text to speech so it can be used later
def text_to_speech(text):
    # Initialize gTTS with the text to convert
    speech = gTTS(text)

    # Save the audio file to a temporary file
    speech_file = 'speech.mp3'
    speech.save(speech_file)

    # Initialize the mixer (pygame audio component)
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load(speech_file)

    # Play the audio file
    pygame.mixer.music.play()

    # Wait while the audio is playing
    while pygame.mixer.music.get_busy():
        time.sleep(2)
    
    # Clean up
    pygame.mixer.quit()

#! /usr/bin/python

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2

username = 0


#Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# initialize the video stream and allow the camera sensor to warm up
# Set the ser to the followng
# src = 0 : for the build in single web cam, could be your laptop webcam
# src = 2 : I had to set it to 2 inorder to use the USB webcam attached to my laptop
vs = VideoStream(src=0,framerate=10).start()
#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)



# loop over frames from the video file stream
while username == 0:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	# Detect the fce boxes
	boxes = face_recognition.face_locations(frame)
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(frame, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown" #if face is not recognized, then print Unknown

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

			#If someone in your dataset is identified, print their name on the screen
			if currentname != name:
				currentname = name
				print(currentname)
				username = currentname
				break

		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)



# do a bit of cleanup
cv2.destroyAllWindows()#!
vs.stop()

# after recognition the gpt api will be loaded

openai.api_key = 'cannot put that on github'


messages = []
system_msg = "you are a voice chat bot capable of engaging in conversation in a very casual and relaxed manner, right now you are talking to" + username + " you have already sad hello to him"
messages.append({'role': 'user', 'content':system_msg})

text_to_speech("Hello " + username + " How are you doing")
print("Assistbot has been loaded! Welcome " + username)
while input != "quit()":
	message = input()
	messages.append({'role': 'user', 'content': message})
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=messages)
	reply = response["choices"][0]["message"]["content"]
	messages.append({"role": "assistant", "content": reply})
	print("\n" + reply + "\n")
	text_to_speech(reply)
	
