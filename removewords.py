import math
import json, jsonlines
import collections
import pickle
from pathlib import Path
from tqdm import tqdm
from utils_wapo import load_clean_wapo_with_embedding
from text_processing import TextProcessing

def idf(N: int, df: int) -> float:
    if df==0:
        return 0
    return math.log10(N / df)

def tf(freq: int) -> float:
    if freq>0:
        return 1+ math.log10(freq)
    else:
        return 0.0

def removestopwords(df_counter, doc_tf_dict, doc_id, content, tfidfscore):
    text_processor = TextProcessing.from_nltk()
    tokens = content.split()
    for i in range(len(tokens) - 1, -1, -1):
        token = tokens[i]
        temptoken = text_processor.normalize(token,True)
        idfscore = idf(len(doc_tf_dict), df_counter[temptoken])
        currenttfidfscore = idfscore * doc_tf_dict[doc_id][temptoken]
        if currenttfidfscore < tfidfscore:
            tokens.pop(i)
    tokens = " ".join(tokens)
    return tokens

text_processor = TextProcessing.from_nltk()
wapo_file = "subset_wapo_50k_sbert_ft_filtered.jl"
pickle_in = open("df_counter.pkl","rb")
df_counter = pickle.load(pickle_in)
pickle_in2= open("doc_tf_dict.pkl","rb")
doc_tf_dict = pickle.load(pickle_in2)
with jsonlines.open('../../../cosi132hw5/pa5_official/pa5_data/50kwithoutterms', mode='a') as writer:
    for i, doc in tqdm(enumerate(load_clean_wapo_with_embedding(wapo_file))):
        title = doc.get("title", "") if doc.get("title") else ""
        doc_id = doc.get("doc_id", "") if doc.get("doc_id") else ""
        author = doc.get("author", "") if doc.get("author") else ""
        published_date = doc.get("published_date", "") if doc.get("published_date") else ""
        content = doc.get("content_str", "") if doc.get("content_str") else ""
        annotation = doc.get("annotation", "") if doc.get("annotation") else ""
        content = removestopwords(df_counter, doc_tf_dict, i, content, .1)
        info = {}
        info["title"] = title
        info["doc_id"]=doc_id
        info["author"] = author
        info["published_date"] = published_date
        info["content"] = content
        info["annotation"] = annotation
        writer.write(info)



