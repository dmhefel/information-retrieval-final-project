# information-retrieval-final-project

# Team 
Team members: Ryan Pyatetsky, Daniel Hefel, Nicole Zamora
Team member submitting code:

# TREC topic

## Number: 439

## TREC Information:
Title: inventions, scientific discoveries 

Description: What new inventions or scientific discoveries have been made?  

Narrative:
The word "new" in the description is defined as occurring in the 1990s. Documents that indicate a "recent" invention or scientific discovery are considered relevant. Discoveries made in astronomy or any scientific discoveries that are not patentable are not relevant. 

## Queries:
 ### From TREC XML 
 
 Title
 Description
 Narrative
 
 ### User Queries:
 Scientific Discovery 
 Scientific Discovery patent 
 Scientific Discovery not astronomy

# Summary: 

SBERT doesn't process the full article, but rather takes the first 500 words. To see if processing the full article would make a difference, the article is segmented into portions and Doc2Vec is trained on the full document. 
Additionally, original articles are pre-processed by deleting the terms that go below a certain threshold, and then tested against the bm25 custom analyzer. 
_(Provide a brief overview of what your team did. This will be described in more detail in the slide presentation, so you don’t need to go into great detail here.)_

# Output:

_Depending on your approach, output may be in the form of tables of results and/or user-visible features of a run-time application._

# Results:
Note: all of our results are compared against the baseline provided by the bm25 custom analyzer.  

The 2-tier search system ignored all of the relevant results that were also found in a query including the word astronomy. This approach performed better in all of the TREC query texts, especially on the narrative (almost doubled the ndcg). 
_Provide a brief overview of the results, which will be described in more detail in the slide presentation._

# Dependencies:
_Describe all resources used for your system, including version numbers and download locations for code obtained from the web._

pip install gensim

pip install -U sentence-transformers

## Data

Download from 
https://drive.google.com/u/0/uc?id=1Y03Cgf-84lua5cmBEt8-4JQKbdgvu3vG&export=download

# Build instructions:
_Describe how to build your system and run it. A commented script or scripts including all steps would suffice._

### Word2Vec

### SBERT

### 2-Tier Search System 
Base code references hw5: 
Run the elasticsearch server on the terminal
Build the wapo_50k_index by reading the 50k json file
Call the evaluate.py file, which applies the bm25 custom analyzer against the user query, and ignores all of the results found in a query composed of the original text plus the word "astronomy"

### tfidf
Create a dictionary for tf per doc and another one for corpus_tf by running the tfidf.py file. This took about 17 minutes to build locally. 

# Team member contributions:

#### Ryan:

#### Daniel:

#### Nicole:
Processing of documents: Tried loading the 50k documents into a shelf that would contain the tf scores of each document, to later calculate the tfidf per token. After multiple tries with building the index, the last attempt resulting in a 36 hour code-run that was stopped by the kernell, I decided to change the approach to load the tf scores into a dictionary and then into a pickle, following the TA's recommendation. 
Developed the 2-tier search system. 

