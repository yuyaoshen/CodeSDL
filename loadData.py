import os
import pandas as pd
import gensim
from gensim.models import Word2Vec

data_path="./data/BCB/"
tokenizationType="basic"
wv_model="SkipGram"
wv_size=100
src_path=data_path+"dataset_"+tokenizationType+"/"
output=tokenizationType
model_path="./models/BCB/"

model=Word2Vec.load(model_path+"wordEmbedding/wv_size-"+str(wv_size)+"/word2vec_"+tokenizationType+"_"+wv_model+".model")
wv=model.wv

files=os.listdir(src_path)
data=pd.DataFrame(columns=['filename','label','wv_sentence'])
count=0
for filename in files:
	label=filename.split('@')[0]
	label=int(label) #BCB数据集需 -1
	#label=0
	with open(src_path+filename,mode='r',encoding='utf8') as f:
		content=f.read()
		content=content.replace("\n","")
		token_stream=content.split(" ")
		matrix=wv[token_stream]
		wv_sentence=matrix.T
		f.close()
	data.loc[count]=[filename,label,wv_sentence]
	count=count+1
data.to_pickle(data_path+"wv_size-"+str(wv_size)+"/dataset_"+output+"_"+wv_model+".pkl")
