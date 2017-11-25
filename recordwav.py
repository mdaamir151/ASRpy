"""Taken from 
	https://people.csail.mit.edu/hubert/pyaudio/#downloads
	with some modifications
"""

import pyaudio
import wave
import array
from sys import byteorder

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
MAX_AMP = 16384

def PredictOnline():

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

	print("Online...")

	while True:
		
		frames = []
		for i in range(40):
			data = stream.read(CHUNK)
			frames.append(data)


	print("Terminated")
	stream.stop_stream()
	stream.close()
	p.terminate()


def normalize(audio16,threshold=400):

	mx = max(abs(i) for i in audio16)
	
	if mx < threshold:	return
	mul_fact = float(MAX_AMP)/mx
	audio16 = [int(x*mul_fact) for x in audio16]
	
	return audio16


def recordAudio():

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

	print("Recording...")
		
	audio16 = array.array('h')

	for i in range(RATE * RECORD_SECONDS // CHUNK):
		chunk16 = array.array('h',stream.read(CHUNK))
		if byteorder == 'big':
			chunk16.byteswap()
		audio16.extend(chunk16)

	print("Finished Recording")
	audio16 = normalize(audio16)
	stream.stop_stream()
	stream.close()
	p.terminate()

	# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	# wf.setnchannels(CHANNELS)
	# wf.setsampwidth(p.get_sample_size(FORMAT))
	# wf.setframerate(RATE)
	# wf.writeframes(frames)
	# wf.close()

recordAudio()