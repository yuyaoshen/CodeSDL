import pandas as pd
import numpy as np
import time,os
import gensim
from gensim.models import Word2Vec
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
	
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
model_path="models/BCB/"
reports_path="results/"
maxSeqLen=1000
modelName="BiLSTM"
hiddenUnits=400

path="./data/BCB/"
tokenizationType="basic"
wv_modelName="SkipGram"
wv_size=100
src_path=path+tokenizationType+"/"
suffix="-0"

wv_model=Word2Vec.load(model_path+"wordEmbedding/wv_size-"+str(wv_size)+"/word2vec_"+tokenizationType+"_"+wv_modelName+".model")
wv=wv_model.wv
CodeSDLmodel=load_model(model_path+modelName+"-"+str(hiddenUnits)+"/wv_size-"+str(wv_size)+"/CodeSDL-"+modelName+"_"+tokenizationType+"_"+wv_modelName+suffix+".h5")

currentTime=time.strftime("%m-%d_%H_%M_%S",time.localtime())
print(currentTime)
print(suffix)
filenameList=pd.DataFrame(columns=['filename'])
predictions=[]
count=0
files=os.listdir(src_path)
for filename in files:
	filenameList.loc[count]=[filename]
	with open(src_path+filename,mode='r',encoding='utf8') as f:
		content=f.read()
		f.close()
	content=content.replace("\n","")
	token_stream=content.split(" ")
	matrix=wv[token_stream]
	sample=pad_sequences([list(zip(*matrix.T))],maxlen=maxSeqLen,dtype='float32',padding='post',truncating='post',value=0.0)
	vector=CodeSDLmodel.predict(sample,verbose=0)[0]
	predictions.append(vector)
	count=count+1
	
predictions=pd.DataFrame(predictions)
reports=pd.concat([filenameList,predictions],axis=1)
currentTime=time.strftime("%m-%d_%H_%M_%S",time.localtime())
print(currentTime)
print(reports)
reports.to_pickle(reports_path+"CodeSDL-"+modelName+"_"+tokenizationType+"_"+wv_modelName+suffix+".pkl")
reports.to_csv(reports_path+"CodeSDL-"+modelName+"_"+tokenizationType+"_"+wv_modelName+suffix+".txt",sep='\t', index=False)

