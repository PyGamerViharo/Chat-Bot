import numpy
import tflearn
import tensorflow
import random
import json
import nltk
import pickle
import os
import time
import playsound
import speech_recognition as sr
from gtts import gTTS
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

while True:
	mode = input("Do you want to chat or speak with the bot? (c/s)")
	if mode == "c":
		with open("intents.json") as file:
			data = json.load(file)

		try:
			with open("data.pickle", "rb") as f:
				words, labels, training, output = pickle.loads(f)

		except:
			words = []
			labels = []
			docs_x = []
			docs_y = []

			for intent in data["intents"]:
				for pattern in intent["patterns"]:
					wrds = nltk.word_tokenize(pattern)
					words.extend(wrds)
					docs_x.append(wrds)
					docs_y.append(intent["tag"])

					if intent["tag"] not in labels:
						labels.append(intent["tag"])

			words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
			words = sorted(list(set(words)))

			labels = sorted(labels)

			training = []
			output = []

			out_empty = [0 for _ in range(len(labels))]

			for x, doc in enumerate(docs_x):
				bag = []

				wrds = [stemmer.stem(w) for w in doc]

				for w in words:
					if w in wrds:
						bag.append(1)

					else:
						bag.append(0)

				output_row = out_empty[:]
				output_row[labels.index(docs_y[x])] = 1

				training.append(bag)

				output.append(output_row)

			training = numpy.array(training)
			output = numpy.array(output)

			with open("data.pickle", "wb") as f:
				pickle.dump((words, labels, training, output), f)

		tensorflow.compat.v1.reset_default_graph()

		net = tflearn.input_data(shape=[None, len(training[0])])
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, (len(output[0])), activation="softmax")
		net = tflearn.regression(net)

		model = tflearn.DNN(net)
		
		"""
		try:
			model.load("model.tflearn")

		except:
			"""
		model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
		model.save("model.tflearn")

		def bag_of_words(s, words):
			bag = [0 for _ in range(len(words))]
			
			s_words = nltk.word_tokenize(s)
			s_words = [stemmer.stem(word.lower()) for word in s_words]

			for se in s_words:
				for i, w in enumerate(words):
					if w == se:
						bag[i] = 1

			return numpy.array(bag)

		def chat():
			print("Bot ready! Start chatting by typing something! To quit, type exit.")
			while True:
				input_var = input("You: ")

				if input_var.lower == "exit":
					quit()
				
				result = model.predict([bag_of_words(input_var, words)])
				results_index = numpy.argmax(result)

				tag = labels[results_index]

				for tg in data["intents"]:
					if tg["tag"] == tag:
						responses = tg["responses"]
				
				print(random.choice(responses))

		chat()

	elif mode == "s":
		with open("intents.json") as file:
			data = json.load(file)

		try:
			with open("data.pickle", "rb") as f:
				words, labels, training, output = pickle.loads(f)

		except:
			words = []
			labels = []
			docs_x = []
			docs_y = []

			for intent in data["intents"]:
				for pattern in intent["patterns"]:
					wrds = nltk.word_tokenize(pattern)
					words.extend(wrds)
					docs_x.append(wrds)
					docs_y.append(intent["tag"])

					if intent["tag"] not in labels:
						labels.append(intent["tag"])

			words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
			words = sorted(list(set(words)))

			labels = sorted(labels)

			training = []
			output = []

			out_empty = [0 for _ in range(len(labels))]

			for x, doc in enumerate(docs_x):
				bag = []

				wrds = [stemmer.stem(w) for w in doc]

				for w in words:
					if w in wrds:
						bag.append(1)

					else:
						bag.append(0)

				output_row = out_empty[:]
				output_row[labels.index(docs_y[x])] = 1

				training.append(bag)

				output.append(output_row)

			training = numpy.array(training)
			output = numpy.array(output)

			with open("data.pickle", "wb") as f:
				pickle.dump((words, labels, training, output), f)

		tensorflow.compat.v1.reset_default_graph()

		net = tflearn.input_data(shape=[None, len(training[0])])
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, (len(output[0])), activation="softmax")
		net = tflearn.regression(net)

		model = tflearn.DNN(net)

		try:
			model.load("model.tflearn")

		except:
			model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
			model.save("model.tflearn")

		def bag_of_words(s, words):
			bag = [0 for _ in range(len(words))]
			
			s_words = nltk.word_tokenize(s)
			s_words = [stemmer.stem(word.lower()) for word in s_words]

			for se in s_words:
				for i, w in enumerate(words):
					if w == se:
						bag[i] = 1

			return numpy.array(bag)

		def chat():
			print("Bot ready! Start by saying somthing!")
			print("NOTE: Pls speak a little slowly and clearly to make sure the VoiceBot can understand you.")
			while True:				
				def get_audio():
					r = sr.Recognizer()
					with sr.Microphone() as source:
						print("You:")
						audio = r.listen(source)
						said = ""
						try:
							said = r.recognize_google(audio)

						except Exception as e:
							print("Exception: " + str(e))

					return said

				ga = get_audio()

				print(str(ga))
				
				result = model.predict([bag_of_words(str(ga), words)])
				results_index = numpy.argmax(result)

				tag = labels[results_index]

				for tg in data["intents"]:
					if tg["tag"] == tag:
						responces = tg["responses"]

				def speak(responces):
					tts = gTTS(text=responces, lang="en")
					filename = "DO_NOT_TOUCH.mp3"
					tts.save(filename)
					playsound.playsound(filename)

				speak(random.choice(responces))

				path = os.getcwd()
				os.remove(path + "\\DO_NOT_TOUCH.mp3")

		chat()
