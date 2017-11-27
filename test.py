

from audio_class import AudioClassifier

files = ['brian.wav','amy.wav','emma.wav','eric.wav','joey.wav','kimberly.wav']
ws = 4096
nfilt = 150
stride = 128
trainTestRatio = 0.8

acl = AudioClassifier(ws=ws,nfilt=nfilt,stride=stride)

train_feat,train_class,test_feat, test_class = acl.trainTestSplit(files)
print("training....")
acl.train(train_feat,train_class)
print("training finished")


print('test data size :',len(test_class))
print("accuracy on test dataset = {} %".format(acl.getAccuracy(test_feat,test_class)))
print("misclassification stat for test set: ",acl.misClassStat(test_feat,test_class))


print("predicting ....")
feat = acl.getnFeatVecs('kimberlyt.wav',n=50)
p = list(acl.predict(feat))

cl = set(p)
# print(cl)
for i in cl:
	print(" {} : {} ".format(files[i],p.count(i)))

# print(list(files[i] for i in p))