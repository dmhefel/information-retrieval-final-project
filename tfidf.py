import collections
import pickle
from pathlib import Path
from tqdm import tqdm
from utils_wapo import load_clean_wapo_with_embedding
from text_processing import TextProcessing

if __name__ == '__main__':
    text_processor = TextProcessing.from_nltk()
    df_counter = collections.Counter()
    doc_tf_dict = {}
    data_dir = Path("data")
    wapo_file = data_dir.joinpath("subset_wapo_50k_sbert_ft_filtered.jl")
    for i, doc in tqdm(enumerate(load_clean_wapo_with_embedding(wapo_file))):
        title = doc.get("title", "") if doc.get("title") else ""
        content = doc.get("content_str", "") if doc.get("content_str") else ""
        tokens = text_processor.get_valid_tokens(title, content, use_stemmer=True)
        tf_dict = collections.Counter(tokens)
        doc_tf_dict[i] = tf_dict
        df_counter.update(tf_dict.keys())

    print("the df counter is", len(df_counter))  # number of terms in the corpus
    print("num docs in corpus", len(doc_tf_dict))  # number of documents in the corpus
    pickle.dump(df_counter, open("df_counter.pkl", "wb"))
    pickle.dump(doc_tf_dict, open("doc_tf_dict.pkl", "wb"))
