import argparse
from typing import List
from metrics import ndcg, Score
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Match, MatchAll, ScriptScore, Ids, Query
from elasticsearch_dsl.connections import connections
from embedding_service.client import EmbeddingClient
from utils import parse_wapo_topics

k = None
id = "439"

def generate_script_score_query(query_vector: List[float], vector_name: str) -> Query:
    """
    generate an ES query that match all documents based on the cosine similarity
    :param query_vector: query embedding from the encoder
    :param vector_name: embedding type, should match the field name defined in BaseDoc ("ft_vector" or "sbert_vector")
    :return: an query object
    """
    q_script = ScriptScore(
        query={"match_all": {}},  # use a match-all query
        script={  # script your scoring function
            "source": f"cosineSimilarity(params.query_vector, '{vector_name}') + 1.0",
            # add 1.0 to avoid negative score
            "params": {"query_vector": query_vector},
        },
    )
    return q_script


def search(index: str, query: Query, k: int, query_text: str) -> List[int]:
    """
    author: Nicole Zamora
    method that applies search to a query based on an specific index
    a secondary query is created by adding the word astronomy to the original query text
    the hits that appear on the secondary query are ignored in the results
    params: index with data, k is the number of docs to be retrieved and query to be used for the search
    return: list with relevance scores of results, list of results
    """
    global id
    s = Search(using="default", index=index).query(query)[:3*k]
    response = s.execute()
    query_astronomy = Match(custom_content={
        "query": query_text + "astronomy"})
    s_astronomy = Search(using="default", index=index).query(query_astronomy)[:k]
    response_astronomy = s_astronomy.execute()

    relevance_lst = []
    results_lst = []
    astronomy_docid_set = set()
    ranking_num = 0
    for hit in response_astronomy:
        astronomy_docid_set.add(hit.doc_id)

    iter = 0
    for hit in response:
        if hit.doc_id not in astronomy_docid_set and iter <=20:
        # if iter <= 20:
            iter +=1
            if hit.annotation == id+"-2":
                relevance_lst.append(2)
            elif hit.annotation ==id+"-1":
                relevance_lst.append(1)
            else:
                relevance_lst.append(0)
            results_lst.append({"date": hit.date,
                                "title": hit.title,
                                "content": hit.content,
                                "author": hit.author,
                                "doc_id": hit.doc_id,
                                "ranking": ranking_num})
            print(
                hit.meta.id, hit.annotation, hit.title, sep="\t"
            )
            ranking_num += 1
    metrics = Score.eval(relevance_lst, k)
    print("NDCG", metrics.ndcg, "\nPrecision",metrics.prec, "\nAvg Precision", metrics.ap)

    return relevance_lst[:k], results_lst[:k]


def search_top_k(index: str, query: Query, k: int) -> List[int]:
    """
    method that finds the top k results of a search
    params: index with data, k is the number of docs to be retrieved and query to be used for the search
    return: list of the top k results
    """
    s = Search(using="default", index=index).query(query)[:k]  # initialize a query and return top five results
    response = s.execute()
    list_docs = []
    for hit in response:
        list_docs.append(hit.meta.id)
    return list_docs


def bm25_default_search(query_text: str, index: str, k: int) -> List[int]:
    """
    method that performs bm25 + default analyzer search
    params: query text, k is the # of docs to be retrieved from the index index
    return: returns the return statement from the search method (relevance score list, and doc results list)
    """
    query = Match( content={"query": query_text} )  # a query that matches "D.C" in the title field of the index, using BM25 as default
    return search(index, query, k)


def bm25_custom_search(query_text: str, index: str, k: int) -> List[int]:
    """
    method that performs bm25 + custom analyzer search
    params: query text, k is the # of docs to be retrieved from the index index
    return: returns the return statement from the search method (relevance score list, and doc results list)
    """
    query = Match( custom_content={"query": query_text} )  # a query that matches query_text in the title field of the index, using BM25 as default
    return search(index, query, k, query_text)



def get_compound_query(query_text_str: str, index: str, q_vector: Query, k: int):
    """
    method that finds the top k docs using bm25+default analyzer, transforms the results into a query
    and makes a compound query with the embedding query
    params: query_text_str is the query text, q_vector is the embedding query, k is the # of docs to be retrieved
    and index is the index name
    return: compound query
    """
    bm25_query = Match(content={"query": query_text_str})
    top_k_ids = search_top_k(index, bm25_query, k)
    q_match_ids = Ids(values=top_k_ids)
    compound_q = (q_vector & q_match_ids)
    return compound_q


def reranking_embedding(index: str, query_text_str: str, q_vec_encoder: List[float], k: int, vector_name: str)-> List[int]:
    """
    generates a script from a query vector based on a specific embedding type (sbert or fastText)
    gets a compound query of that vector and the top k docs using bm25+default analyzer
    params: index is the index name, query_text_str is the query, q_vec_encoder is the query vector,
    vector name is the embedding type and k is the number of docs to be retrieved
    return: bm25+default analyzer search result reranked by an embedding type
    """
    q_vector = generate_script_score_query(q_vec_encoder, vector_name)
    compound_query = get_compound_query(query_text_str, index, q_vector, k)
    return search(index, compound_query, k)


def main():
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_name", required=True, type=str, help="name of the ES index"
    )
    parser.add_argument(
        "--topic_id", required=True, type=str, help="document topic id",
    )
    parser.add_argument(
        "--query_type", required=True, type=str, help="document section for query: title, description or narration"
    )
    parser.add_argument(
        "-u", required=False, type=str, help="yes for custom analyzer no for default"
    )
    parser.add_argument(
        "--vector_name", required=False, type=str, help="Embedding vector name"
    )
    parser.add_argument(
        "--top_k", required=True, type=int, help="tops k document ranking"
    )
    args = parser.parse_args()
    global k
    k = args.top_k
    query_txt = []
    topic_mapping = parse_wapo_topics("pa5_data/topics2018.xml")
    idx_name = args.index_name
    query_type = args.query_type
    if query_type == "title":
        query_txt.append(topic_mapping[args.topic_id][0])
    elif query_type == "description":
        query_txt.append(topic_mapping[args.topic_id][1])
    else:
        query_txt.append(topic_mapping[args.topic_id][2])
    print(query_txt, "\n")
    if args.vector_name:
        v_name = args.vector_name
        if v_name == "sbert_vector":
            encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
        else:
            encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")  # connect to the fasttext embedding server
        q_vec_encoder = encoder.encode(query_txt, pooling="mean").tolist()[0]
        relevance_scores, results_dict = reranking_embedding(idx_name, query_txt[0], q_vec_encoder, k, v_name)

    else:
        if args.u == "yes":
            relevance_scores, results_dict = bm25_custom_search(query_txt[0], idx_name, k)
        else:
            relevance_scores, results_dict = bm25_default_search(query_txt[0], idx_name, k)

if __name__ == "__main__":
    main()



