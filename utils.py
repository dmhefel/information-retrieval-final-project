from pathlib import Path
import json

def load_wapo(wapo_jl_path):
    """
    It should be similar to the load_wapo in HW3 with two changes:
    - for each yielded document dict, use "doc_id" instead of "id" as the key to store the document id.
    - convert the value of "published_date" to a readable format e.g. 2021/3/15. You may consider using python datatime package to do this.
    """
    # open the jsonline file
    with open(wapo_jl_path) as f:
        # get line
        line = f.readline()
        # id number
        id = 0
        # while line not empty
        while line:
            # load line as json
            article = json.loads(line)
            # process content by getting each sanitized html sentence and adding to list.
            # content = []
            # for sentence in article['contents']:
            #     if sentence is not None and 'content' in sentence and sentence['type'] == 'sanitized_html':
            #         content.append(sentence['content'])
            # # turn list of sentences into one string
            # content = ' '.join(content)
            # # use regex to get rid of html elements
            # content = re.sub('<[^<]+?>', '', content)
            #process date
            #dateinfo = dt.fromtimestamp(article['published_date'] / 1000)
            #date = str(dateinfo.year) + "/" + str(dateinfo.month) + "/" + str(dateinfo.day)

            date = article['published_date']
            # create dict for the article
            articledict = {'id': id,
                           'title': article["title"] if article['title'] is not None else '',
                           "author": article["author"] if article['author'] is not None else '',
                           "published_date": date,
                           "content_str": article['content_str']}
            # increment id #
            id += 1
            # get next line
            line = f.readline()
            # yield the dictionary (generator object)
            yield articledict



data_dir = Path("data")
wapo_path = data_dir.joinpath("50k.jl")
documents = [doc['content_str'] for doc in load_wapo(wapo_path)]

print(documents[0])











from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

data = ["I love machine learning. It's awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]

data = documents

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

#print(tagged_data)

max_epochs = 10
vec_size = 50
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=max_epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")



from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("inventions scientific discovery".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.dv.most_similar('1')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.dv['1'])
