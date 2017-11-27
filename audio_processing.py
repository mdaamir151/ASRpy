
import numpy as np

def silent(frame,thresh = 4000):
	assert len(frame.shape) == 1, "multidimensional frame in silent()"
	return frame.max() < thresh


def blocks(data, ws=256, stride=128, rem_silence=True, n=None):

	MAX_AMP = 16384.0

	assert (type(data).__module__ == np.__name__) and (data.dtype == 'float64'), "audio must have float64 type"

	if len(data.shape) > 1:
		print(" @{}-channel audio. Casting into mono channel...".format(len(data.shape)))
		data = data[:,0]
	
	assert ws % 2 == 0, "window size should be even"
	assert stride > 0, "stride cannot be zero"
	assert data.shape[0] >= ws , "insufficient data"

	# normalize volume
	mx = data.max() + 1e-6
	fact = MAX_AMP / mx
	data = data * fact

	sz = len(data) - ws 
	nw = sz // stride + 1

	frame = np.empty((nw,ws), dtype = data.dtype)

	k = 0

	for i in range(nw):

		start = i * stride
		end = start + ws
		f = data[start:end]
		
		if rem_silence and silent(f):	pass
		else :
			frame[k]=f 
			k += 1

		if n and n == k:	break

	if n:	
		assert n == k,"value of param 'n' cannot exceed {} frames".format(k)

	frame.resize((k,ws))
	# print(frame.shape)

	return frame



def stft(data, ws = 256, stride = 128, mean_norm = True, real = True, window = np.hamming, n=None):

	assert (type(data).__module__ == np.__name__) and (data.dtype == 'float64'), "audio must have float64 type"


	if real:
		fft = np.fft.rfft
		# even ws ( sample size ) results in realfft of size ws/2 + 1
		last = -1

	else:
		fft = np.fft.fft
		last = None
	
	if mean_norm :
		data -= data.mean()

	X = blocks(data,ws,stride,n=n)

	X = X * window(ws)

	#result of rfft has size ws/2 + 1
	result = fft(X)[:,:last]

	return result


def spectral_power(datavec, ws = 256, stride = 128, real = True, threshold = 1e-6, norm = True, log = True, window=np.hamming,n=None):

	assert (type(datavec).__module__ == np.__name__) and (datavec.dtype == 'float64'), "audio must have float64 type"
	
	complexSpec = stft(datavec, ws, stride, mean_norm = True, real=real, window=window, n=n)

	#calculate amplitude from complex representation
	spec = np.abs(complexSpec)

	if norm:
		mx = spec.max(axis=1)
		mx.shape = (mx.shape[0],1)

		#normalize amplitude
		spec /= mx  

		spec[spec < threshold] = threshold

	if log:
		spec = np.log10(spec)

	return spec

"""
numpy.fft.rfft or numpy.fft.fft of sample has frequencies k/NT where N = window size, T = time period 
of sampling. So the frequecies are 0, 1/NT, 2/NT, 3/NT ....
fft has length equal to half the window size 

when ws is even and k = ws/2:
	size of numpy.fft.fft is ws/2 + 1
	freq = (ws/2 * samprate)/ws , N = ws, i.e. window size
	therefore freq[ws/2] = samprate/2
	ws/2 is the (ws/2 + 1)th index of the fft bins, since it is 0-indexed

	fr[i] = i / NT = i*samprate/ws

	i = np.floor(fr*ws/samprate)
"""

def mel_bins(nfilt = 30, ws = 256, samprate = 12500, lowfreq = 100, highfreq = None):

	highfreq = highfreq or samprate / 2
	assert highfreq <= samprate / 2, "highfreq must not exceed half of samprate"

	lowmel = 2595 * np.log10(1 + lowfreq / 700.)
	highmel = 2595 * np.log10(1 + highfreq / 700.)

	melpoints = np.linspace(lowmel,highmel,nfilt+2)

	hzpoints = 700 * (10 ** (melpoints / 2595.0) - 1) + 1e-6

	bins = np.floor(hzpoints * ws / samprate)
	bins = bins.astype(int)

	return bins


def mel_filter_power(powerspec, bins):

	nsamples = powerspec.shape[0]

	nfilt = bins.shape[0] - 2

	filter_energies = np.zeros((nsamples,nfilt),dtype='float64')


	for i in range(1,nfilt+1):
		for j in range(bins[i-1],bins[i]):
			w = (j-bins[i-1])/(bins[i]-bins[i-1])
			filter_energies[:,i-1] += powerspec[:,j] * w

		filter_energies[:,i-1] += powerspec[:,bins[i]]

		for j in range(bins[i]+1,bins[i+1]):
			w = (bins[i+1]-j)/(bins[i+1]-bins[i])
			filter_energies[:,i-1] += powerspec[:,j] * w

	return filter_energies

