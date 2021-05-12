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
 
_If you make any changes to the title, description, and narrative, or create other “user queries”, describe them here. Give each query type a unique name so that they can be referred to when describing what you did and the output._

# Summary: 

SBERT doesn't process the full article, but rather takes the first 500 words. To see if processing the full article would make a difference, the article is segmented into portions and Doc2Vec is trained on the full document. 
Additionally, original articles are pre-processed by deleting the terms that go below a certain threshold, and then tested against the bm25 custom analyzer. 
_(Provide a brief overview of what your team did. This will be described in more detail in the slide presentation, so you don’t need to go into great detail here.)_

# Output:
_Depending on your approach, output may be in the form of tables of results and/or user-visible features of a run-time application._

# Results:
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

# Team member contributions:

Ryan:

Daniel:

Nicole:

1st iteration was 4 minutes per epoch 
