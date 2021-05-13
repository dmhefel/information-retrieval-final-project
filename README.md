# information-retrieval-final-project

# Team 
Team members: Ryan Pyatetsky, Daniel Hefel, Nicole Zamora
Team member submitting code:

=========================
# TREC topic

## Number: 439

## TREC Information:
Title: inventions, scientific discoveries 

Description: What new inventions or scientific discoveries have been made?  

Narrative:
The word "new" in the description is defined as occurring in the 1990s. Documents that indicate a "recent" invention or scientific discovery are considered relevant. Discoveries made in astronomy or any scientific discoveries that are not patentable are not relevant. 

## Queries:
 ### From TREC XML 
 
 * Title
 * Description
 * Narrative
 
 ### User Queries:
 * Scientific Discovery 
 * Scientific Discovery patent 
 * Scientific Discovery not astronomy
 
=========================
# Summary: 
We took several approaches to the document retrieval task:

SBERT doesn't process the full article, but rather takes the first 500 or so words. This means that if relevant information is at the end of the document, it is not included. To see if processing the full article would make a difference, the article is segmented into portions, vectors created for each portion, and then the embeddings are averaged among portions.

 We also trained a doc2vec model on the corpus to create document vectors that could be used for retrieval.

Additionally, original articles are pre-processed by deleting the terms that go below a certain threshold, and then tested against the bm25 custom analyzer. 


=========================
# Output:

_Depending on your approach, output may be in the form of tables of results and/or user-visible features of a run-time application._

=========================
# Results:
Note: all of our results are compared against the baseline provided by the bm25 custom analyzer.  

The 2-tier search system ignored all of the relevant results that were also found in a query including the word astronomy. This approach performed better in all of the TREC query texts, especially on the narrative (almost doubled the ndcg). 
_Provide a brief overview of the results, which will be described in more detail in the slide presentation._

=========================
# Dependencies:
_Describe all resources used for your system, including version numbers and download locations for code obtained from the web._

pip install gensim

pip install -U sentence-transformers

## Data

Download from 
https://drive.google.com/u/0/uc?id=1Y03Cgf-84lua5cmBEt8-4JQKbdgvu3vG&export=download

=========================

# Build instructions:
_Describe how to build your system and run it. A commented script or scripts including all steps would suffice._

The Word2Vec, retraining of SBERT and 2-tier search system required the use of some code from hw5. So, we have attached zip files to the projects (pa5_official_nicolezamora.zip and ------), and the files that we modified can be found on the main directory of this repository. 

### doc2Vec
-run the doc2vec.py file and then use the resulting jl file to build the elastic search index. See the query.py file for how to run queries with elastic search

### SBERT
-run the sbert.py file and then use the results jl file to build the elastic search index by adjusting the basedoc to take in the new vector

### 2-Tier Search System 
* Base code references hw5: 
* Run the elasticsearch server on the terminal
* Build the wapo_50k_index by reading the 50k json file
* Call the evaluate.py file, which applies the bm25 custom analyzer against the user query, and ignores all of the results found in a query composed of the original * text plus the word "astronomy"

### tfidf
Create a dictionary for tf per doc and another one for corpus_tf by running the tfidf.py file. This took about 17 minutes to build locally. 

=========================
# Team member contributions:

#### Ryan:

#### Daniel: 
Wrote and trained the doc2vec and sbert approaches and their evaluation file. List of .py files written by Daniel:
utils.py
sbert.py
doc2vec.py
query.py
One version of doc_template.py and index.py
All files located in the eponymous zip file


#### Nicole:
Processing of documents: Tried loading the 50k documents into a shelf that would contain the tf scores of each document, to later calculate the tfidf per token. After multiple tries with building the index, the last attempt resulting in a 36 hour code-run that was stopped by the kernel, I decided to change the approach to load the tf scores into a dictionary and then into a pickle, following the TA's recommendation. 
Developed the 2-tier search system. 

