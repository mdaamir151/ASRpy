

from audio_class import AudioClassifier

files = ['aariz.wav','english.wav','vf3-08.wav','woman.wav','shab.wav','maaroof.wav']
ws = 4096
nfilt = 120
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
feat = acl.getnFeatVecs('marooft.wav',n=20)
p = acl.predict(feat)
print(list(files[i] for i in p))