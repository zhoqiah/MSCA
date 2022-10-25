# 开发团队：万翼科技
# 开发人员： jony
# 开放时间：  年 月
# 文件名称： name.py
# 开发工具： pycharm

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-uncased/')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")