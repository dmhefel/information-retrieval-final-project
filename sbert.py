#pip install -U sentence-transformers
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('msmarco-distilbert-base-v3')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings of type ndarray and shape (768,)
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
    print(embedding.shape)
print(len(embeddings))
#seeing original max which is 510
print("Max Sequence Length:", model.max_seq_length)
#setting it to something else. Maybe not needed?
#model.max_seq_length = 200
#print("Max Sequence Length:", model.max_seq_length)




query_embedding = model.encode('How big is London')
passage_embedding = model.encode('London has 9,787,426 inhabitants at the 2011 census')

print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))