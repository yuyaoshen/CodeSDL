import gensim
from gensim.models import Word2Vec

tokenizationType="basic"
dataset="./data/BCB/doc_"+tokenizationType+".txt"
model_path="./models/BCB/"
wv_size=100
windowSize=5
epochsSize=50

src_stream=open(dataset, encoding="utf8")
data=[]
line=src_stream.readline()
while line:
	line=line.replace("\n","")
	temp=line.split(" ")
	data.append(temp)
	line=src_stream.readline()

#wv_model="CBOW"
#model = gensim.models.Word2Vec(data, min_count = 1, vector_size = wv_size, window = windowSize, epochs=epochsSize, workers=6)
#model.save(model_path+"wordEmbedding/wv_size-"+str(wv_size)+"/word2vec_"+tokenizationType+"_"+wv_model+".model")

wv_model="SkipGram"
model = gensim.models.Word2Vec(data, min_count = 1, vector_size = wv_size, window = windowSize, epochs=epochsSize, workers=6, sg = 1)
model.save(model_path+"wordEmbedding/wv_size-"+str(wv_size)+"/word2vec_"+tokenizationType+"_"+wv_model+".model")

