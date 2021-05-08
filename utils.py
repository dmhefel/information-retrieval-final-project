from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('msmarco-distilbert-base-v3') #encoder for sbert
#global variable for determing how long the text for chunking and going into sbert encoder
CHUNKSIZE = 300



jane_austen = '''Darcy only smiled; and the general pause which ensued made
      Elizabeth tremble lest her mother should be exposing herself
      again. She longed to speak, but could think of nothing to say;
      and after a short silence Mrs. Bennet began repeating her thanks
      to Mr. Bingley for his kindness to Jane, with an apology for
      troubling him also with Lizzy. Mr. Bingley was unaffectedly civil
      in his answer, and forced his younger sister to be civil also,
      and say what the occasion required. She performed her part indeed
      without much graciousness, but Mrs. Bennet was satisfied, and
      soon afterwards ordered her carriage. Upon this signal, the
      youngest of her daughters put herself forward. The two girls had
      been whispering to each other during the whole visit, and the
      result of it was, that the youngest should tax Mr. Bingley with
      having promised on his first coming into the country to give a
      ball at Netherfield.

      Lydia was a stout, well-grown girl of fifteen, with a fine
      complexion and good-humoured countenance; a favourite with her
      mother, whose affection had brought her into public at an early
      age. She had high animal spirits, and a sort of natural
      self-consequence, which the attention of the officers, to whom
      her uncle’s good dinners, and her own easy manners recommended
      her, had increased into assurance. She was very equal, therefore,
      to address Mr. Bingley on the subject of the ball, and abruptly
      reminded him of his promise; adding, that it would be the most
      shameful thing in the world if he did not keep it. His answer to
      this sudden attack was delightful to their mother’s ear:

      “I am perfectly ready, I assure you, to keep my engagement; and
      when your sister is recovered, you shall, if you please, name the
      very day of the ball. But you would not wish to be dancing when
      she is ill.”

      Lydia declared herself satisfied. “Oh! yes—it would be much
      better to wait till Jane was well, and by that time most likely
      Captain Carter would be at Meryton again. And when you have given
      _your_ ball,” she added, “I shall insist on their giving one
      also. I shall tell Colonel Forster it will be quite a shame if he
      does not.”

      Mrs. Bennet and her daughters then departed, and Elizabeth
      returned instantly to Jane, leaving her own and her relations’
      behaviour to the remarks of the two ladies and Mr. Darcy; the
      latter of whom, however, could not be prevailed on to join in
      their censure of _her_, in spite of all Miss Bingley’s witticisms
      on _fine eyes_.




      Chapter 10

      The day passed much as the day before had done. Mrs. Hurst and
      Miss Bingley had spent some hours of the morning with the
      invalid, who continued, though slowly, to mend; and in the
      evening Elizabeth joined their party in the drawing-room. The
      loo-table, however, did not appear. Mr. Darcy was writing, and
      Miss Bingley, seated near him, was watching the progress of his
      letter and repeatedly calling off his attention by messages to
      his sister. Mr. Hurst and Mr. Bingley were at piquet, and Mrs.
      Hurst was observing their game.

      Elizabeth took up some needlework, and was sufficiently amused in
      attending to what passed between Darcy and his companion. The
      perpetual commendations of the lady, either on his handwriting,
      or on the evenness of his lines, or on the length of his letter,
      with the perfect unconcern with which her praises were received,
      formed a curious dialogue, and was exactly in union with her
      opinion of each.

      “How delighted Miss Darcy will be to receive such a letter!”

      He made no answer.

      “You write uncommonly fast.”

      “You are mistaken. I write rather slowly.”

      “How many letters you must have occasion to write in the course
      of a year! Letters of business, too! How odious I should think
      them!”

      “It is fortunate, then, that they fall to my lot instead of
      yours.”

      “Pray tell your sister that I long to see her.”

      “I have already told her so once, by your desire.”

      “I am afraid you do not like your pen. Let me mend it for you. I
      mend pens remarkably well.”

      “Thank you—but I always mend my own.”

      “How can you contrive to write so even?”

      He was silent.

      “Tell your sister I am delighted to hear of her improvement on
      the harp; and pray let her know that I am quite in raptures with
      her beautiful little design for a table, and I think it
      infinitely superior to Miss Grantley’s.”

      “Will you give me leave to defer your raptures till I write
      again? At present I have not room to do them justice.”

      “Oh! it is of no consequence. I shall see her in January. But do
      you always write such charming long letters to her, Mr. Darcy?”

      “They are generally long; but whether always charming it is not
      for me to determine.”

      “It is a rule with me, that a person who can write a long letter
      with ease, cannot write ill.”
'''

#function that chunks the text
def chunk_text(doc):
    #doc is the string representing the text of document
    tokens = doc.split()
    chunked_tokens = [tokens[i:i + CHUNKSIZE] for i in range(0, len(tokens), CHUNKSIZE)]
    return [" ".join(i) for i in chunked_tokens]


#chunk text
chunked_text = chunk_text(jane_austen)
#get the sbert vectors for chunks
embeddings = model.encode(chunked_text)
# embeddings = []
# for chunk in chunked_text:
#     embeddings.append(model.encode(chunk))
#embeddings = np.arange(2, 11).reshape(3,3) #this gives array of [[2,3,4],[5,6,7],[8,9,10]]
#print(len(embeddings))
#print(embeddings)
#add all embeddings
averaged_embedding = embeddings[0]
#print(averaged_embedding)
if len(embeddings)>1:
    for i in range(1,len(embeddings)):
        averaged_embedding += embeddings[i]
#print(averaged_embedding)
averaged_embedding=averaged_embedding/len(embeddings)
print(averaged_embedding)



# def load_wapo(wapo_jl_path):
#     """
#     It should be similar to the load_wapo in HW3 with two changes:
#     - for each yielded document dict, use "doc_id" instead of "id" as the key to store the document id.
#     - convert the value of "published_date" to a readable format e.g. 2021/3/15. You may consider using python datatime package to do this.
#     """
#     # open the jsonline file
#     with open(wapo_jl_path) as f:
#         # get line
#         line = f.readline()
#         # id number
#         id = 0
#         # while line not empty
#         while line:
#             # load line as json
#             article = json.loads(line)
#             # process content by getting each sanitized html sentence and adding to list.
#             # content = []
#             # for sentence in article['contents']:
#             #     if sentence is not None and 'content' in sentence and sentence['type'] == 'sanitized_html':
#             #         content.append(sentence['content'])
#             # # turn list of sentences into one string
#             # content = ' '.join(content)
#             # # use regex to get rid of html elements
#             # content = re.sub('<[^<]+?>', '', content)
#             #process date
#             #dateinfo = dt.fromtimestamp(article['published_date'] / 1000)
#             #date = str(dateinfo.year) + "/" + str(dateinfo.month) + "/" + str(dateinfo.day)
#
#             date = article['published_date']
#             # create dict for the article
#             articledict = {'id': id,
#                            'title': article["title"] if article['title'] is not None else '',
#                            "author": article["author"] if article['author'] is not None else '',
#                            "published_date": date,
#                            "content_str": article['content_str']}
#             # increment id #
#             id += 1
#             # get next line
#             line = f.readline()
#             # yield the dictionary (generator object)
#             yield articledict
#
#
#
# data_dir = Path("data")
# wapo_path = data_dir.joinpath("50k.jl")
# documents = [doc['content_str'] for doc in load_wapo(wapo_path)]
#
# print(documents[0])
#
#
#
#
#
#
#
#
#
#
#
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from nltk.tokenize import word_tokenize
#
# data = ["I love machine learning. It's awesome.",
#         "I love coding in python",
#         "I love building chatbots",
#         "they chat amagingly well"]
#
# data = documents
#
# tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
#
# #print(tagged_data)
#
# max_epochs = 10
# vec_size = 50
# alpha = 0.025
#
# model = Doc2Vec(vector_size=vec_size,
#                 alpha=alpha,
#                 min_alpha=0.00025,
#                 min_count=1,
#                 dm=1)
#
# model.build_vocab(tagged_data)
#
# for epoch in range(max_epochs):
#     print('iteration {0}'.format(epoch))
#     model.train(tagged_data,
#                 total_examples=model.corpus_count,
#                 epochs=max_epochs)
#     # decrease the learning rate
#     model.alpha -= 0.0002
#     # fix the learning rate, no decay
#     model.min_alpha = model.alpha
#
# model.save("d2v.model")
# print("Model Saved")
#
#
#
# from gensim.models.doc2vec import Doc2Vec
#
# model= Doc2Vec.load("d2v.model")
# #to find the vector of a document which is not in training data
# test_data = word_tokenize("inventions scientific discovery".lower())
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)
#
# # to find most similar doc using tags
# similar_doc = model.dv.most_similar('1')
# print(similar_doc)
#
#
# # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
# print(model.dv['1'])
