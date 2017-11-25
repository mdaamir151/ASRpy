import audio_processing as fn
import scipy.io.wavfile as wav
import numpy as np
from sklearn.neural_network import MLPClassifier
import wave


class AudioClassifier:
	
	def __init__(self,ws=256,nfilt=30):
		self.__ws__ = ws
		self.__nfilt__ = nfilt
		self.__highfreq__ = None

	def read(self,file):
		(rate,sig) = wav.read(file)
		sig = sig.astype('float64')
		return rate, sig

	def minframerate(self, files):
		"""returns min sampling frequency of all the audio files 
		power analysis should be done between same set of frequencies """
		samprate = [0]*len(files)

		for i,f in enumerate(files):
			stream = wave.open(f,'r')
			samprate[i] = stream.getparams().framerate
			stream.close()

		minfr = min(samprate)

		return minfr

	def SpecPowerVec(self, sig, samprate, n=None):
		pspec = fn.spectral_power(sig, ws=self.__ws__, n=n)
		bins = fn.mel_bins(nfilt=self.__nfilt__, samprate=samprate,ws=self.__ws__, highfreq=self.__highfreq__)
		pr = fn.mel_filter_power(pspec,bins)
		return pr,len(pspec)

	def trainTestSplit(self,files,ratio=0.8,n=None,highfreq=None):
		if highfreq:
			self.__highfreq__ = highfreq
		else: 
			self.__highfreq__ = self.minframerate(files) // 2
		
		print("high freq limit set at {} Hz".format(self.__highfreq__))
		
		print("processing {} ('{}') ...".format(1,files[0]))
		(rate,sig) = self.read(files[0])
		feat,c = self.SpecPowerVec(sig,rate,n=n)
		y = [0]*c
		print("processed {} ('{}') with {} samples ".format(1,files[0],c))

		for (i,f) in enumerate(files[1:]):
			print("processing {} ('{}') ...".format(i+2,f))
			(rate,sig) = self.read(f)
			featvec,c = self.SpecPowerVec(sig,rate,n=n)
			feat = np.concatenate((feat,featvec))
			y.extend([i+1]*c)	# i starts with 0
			print("processed {} ('{}') with {} samples ".format(i+2,f,c))
		
		y= np.array(y)
		c = y.shape[0]
		r = np.random.permutation(range(c))
		new_feat = feat[r]
		new_y = y[r]
		cut = int(ratio * c)

		train_feat = new_feat[:cut]
		train_class = new_y[:cut]

		test_feat = new_feat[cut:]
		test_class = new_y[cut:]

		return train_feat, train_class, test_feat, test_class

	def train(self,feat,targClass):
		self.__clf__ = MLPClassifier(solver='lbfgs', alpha = 0.000001, random_state=1)
		self.__clf__.fit(feat,targClass)

	def predict(self,feat):
		return self.__clf__.predict(feat)

	def getAccuracy(self,testFeat,testClass):
		c = 0
		p = self.predict(testFeat)
		
		for (r,e) in zip(p,testClass):
			if r == e:
				c += 1
		
		return c*100./len(testClass)

	def getnFeatVecs(self, file, n=1):
		(rate,sig) = self.read(file)
		feat,c = self.SpecPowerVec(sig,rate,n=n)
		return feat

	def misClassStat(self, testvec, targClass):
		p = self.predict(testvec)
		c = len(set(targClass))
		d = {}
		for i in range(c):
			d[i] = 0

		for i,j in zip(p,targClass):
			if i !=j :
				d[j] += 1

		return d


#train with same no. of frames for better performance, else the one greater in number dominates
#all the audio data must have same sampling rate for feature vector to span on same set of frequencies or must have same highest frequency 