import pandas as pd
import numpy as np
import time,os
import warnings
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking,Bidirectional,Dropout,Dense,Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

def serializeMatrix(wv_sentence):
	num=len(wv_sentence)
	i=0
	while i<num:
		wv_sentence[i]=list(zip(*wv_sentence[i]))
		i=i+1

maxSeqLen=1000
classNum=10 #BCB数据集为10类，OJ数据集为104类

hiddenUnits=400
dropout=0.2
batchSize=16
MaxEpoch=30

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

path="./"
tokenizationType="basic"
wv_model="SkipGram"
wv_size=100

train_samples=pd.read_pickle(path+"Samples-train_"+tokenizationType+"_"+wv_model+".pkl")
serializeMatrix(train_samples['wv_sentence'])
train_x=train_samples['wv_sentence']
train_x=pad_sequences(train_x,maxlen=maxSeqLen,dtype='float32',padding='post',truncating='post',value=0.0)
train_y=train_samples['label']
train_y=np.array(train_y).astype(int)
train_y.reshape(len(train_y),1)
train_y=keras.utils.to_categorical(train_y)
del train_samples


dev_samples=pd.read_pickle(path+"Samples-dev_"+tokenizationType+"_"+wv_model+".pkl")
serializeMatrix(dev_samples['wv_sentence'])
dev_x=dev_samples['wv_sentence']
dev_x=pad_sequences(dev_x,maxlen=maxSeqLen,dtype='float32',padding='post',truncating='post',value=0.0)
dev_y=dev_samples['label']
dev_y=np.array(dev_y).astype(int)
dev_y.reshape(len(dev_y),1)
dev_y=keras.utils.to_categorical(dev_y)
del dev_samples

test_samples=pd.read_pickle(path+"Samples-test_"+tokenizationType+"_"+wv_model+".pkl")
serializeMatrix(test_samples['wv_sentence'])
test_x=test_samples['wv_sentence']
test_x=pad_sequences(test_x,maxlen=maxSeqLen,dtype='float32',padding='post',truncating='post',value=0.0)
test_y=test_samples['label']
test_y=np.array(test_y).astype(int)
test_y.reshape(len(test_y),1)
test_y=keras.utils.to_categorical(test_y)
del test_samples


model=Sequential([
	Bidirectional(LSTM(hiddenUnits, return_sequences=False), input_shape=(maxSeqLen,wv_size)),
	Dropout(dropout),
	Dense(classNum+1),
	Activation('softmax'),
])

lr_schedule=keras.optimizers.schedules.ExponentialDecay(
	initial_learning_rate=0.01,
	decay_steps=5,
	decay_rate=0.1)
optimizer=keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#model.fit(x=train_x,y=train_y,epochs=MaxEpoch,shuffle=True,batch_size=batchSize)
model.fit(x=train_x,y=train_y,validation_data=(dev_x, dev_y),epochs=MaxEpoch,shuffle=True,batch_size=batchSize,validation_batch_size=batchSize)
model.evaluate(test_x,test_y,batch_size=batchSize)
currentTime=time.strftime("%m-%d_%H_%M",time.localtime())
print("Saving model to: CodeSDL-BiLSTM_"+tokenizationType+"_"+wv_model+"_"+currentTime+".h5")
model.save(path+"CodeSDL-BiLSTM_"+tokenizationType+"_"+wv_model+"_"+currentTime+".h5")
