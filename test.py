import math
from text_processing import TextProcessing
    @staticmethod
    def idf(N: int, df: int) -> float:
        """
        compute the logarithmic (base 2) idf score
        :param N: document count N
        :param df: document frequency
        :return:
        """
        return math.log10(N/df)

    @staticmethod
    def tf(freq: int) -> float:
        """
        compute the logarithmic tf (base 2) score
        :param freq: raw term frequency
        :return:
        """
        if freq>0:
            return 1+ math.log10(freq)
        else:
            return 0.0



    def removestopwords(termshelf,doc,tfidfscore):
    	text_processor = TextProcessing.from_nltk()
    	tokens = v.split()
    	for i in range(len(tokens)-1,-1,-1):
   			token=tokens[i]
    		temptoken=text_processor.normalize(token)
    		idfscore=idf(50000,len(termshelf[temptoken]))
    		currenttfidfscore=idfscore*termshelf[temptoken][1]
    		if currenttfidfscore<tfidfscore:
    			tokens.pop(i)
    	tokens=" ".join(tokens)
    	return tokens

for document in docs:
    docid = document["doc_id"]
    tokens = text_processor.get_normalized_tokens(document["title"], document["content"])
    temp = dict()
    termtfs = []
    for token in tokens:
        if token in temp:
            temp[token] += 1.0
        else:
            temp[token] = 1.0
    for k, v in temp.items():
        termfreq = text_processor.tf(v)
        termtfs.append(termfreq)
        if k in postingdict:
            # temp=postingdict[k]
            # temp.append((docid,termfreq))
            # postingdict[k]=temp
            postingdict[k].append((docid, termfreq))
        else:
            postingdict[k] = [(docid, termfreq)]


