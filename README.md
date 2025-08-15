# CodeSDL

### Requirements

+ codeprep https://github.com/giganticode/codeprep 需要安装不高于python-3.9版本
+ pandas 1.5.3
+ numpy 1.26.4
+ gensim 4.3.3
+ tensorflow >=2.7.0
+ keras >=2.7.0
+ RAM 16GB or more
+ GPU with CUDA support is also needed
+ BATCH_SIZE should be configured based on the GPU memory size

### Usage

| 文件              | 用途                                                         |
| :---------------- | :----------------------------------------------------------- |
| checkVersion.py | 用于检测环境是否一致，避免包依赖产生的错误。 |
| preprocessing.sh   | 用于处理源代码token流，依赖codeprep项目，可使用basic或bpe选项进行切词。   |
| gatherTokens.sh   | 用于收集数据集中所有tokens，将分词后所有代码tokens集中到同一文本中，并应用于词向量训练。   |
| wordVector.py | 根据所有源代码token流的数据集，训练word embedding模型，用于生成word2vec向量。 |
| loadata.py | 将经过切词的token流载入数据集。 |
| splitData.py | 将有标记数据划分为训练集、验证集和测试集，默认比例为6:2:2。 |
| trainingModel.py | 训练CodeSDL序列分类模型，提取源代码token流中的自然语言语义。 |
| prediction.py | 使用预训练的CodeSDL序列模型对源代码token流进行序列分类，生成源代码语义功能性概率向量。 |
| autoExperiment.sh | 自动进行多次模型训练以实现交叉验证，每次重新分割数据集，并重新训练模型。 |

### Main Steps
1. 使用preprocessing.sh将源代码进行分词，将源代码文本转变为token词序列；
2. 使用word2ec方法（wordVector.py程序），将所有源代码token流进行词嵌入，生成word2vec模型，用于将源代码token序列转换为词向量序列；
3. 使用trainingModel.py进行序列模型的训练，使用prediction.py进行新代码的语义概率向量生成。
