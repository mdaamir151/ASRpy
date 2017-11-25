

from audio_class import AudioClassifier

files = ['shab.wav','vf3-06.wav','english.wav']
ws = 1024
nfilt = 60
trainTestRatio = 0.8

acl = AudioClassifier(ws=ws,nfilt=nfilt)

train_feat,train_class,test_feat, test_class = acl.trainTestSplit(files)
acl.train(train_feat,train_class)

feat = acl.getnFeatVecs('shabtest.wav',n=20)
p = acl.predict(feat)

print(p)

print("misclassification stat for test set: ",acl.misClassStat(test_feat,test_class))

print("accuracy on test dataset = {} %".format(acl.getAccuracy(test_feat,test_class)))