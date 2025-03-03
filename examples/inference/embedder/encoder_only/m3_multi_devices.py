import os
from FlagEmbedding_Aizip import BGEM3FlagModel


def test_m3_multi_devices():
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        devices=["cuda:0", "cuda:1"],   # if you don't have GPUs, you can use ["cpu", "cpu"]
        pooling_method='cls',
        cache_dir=os.getenv('HF_HUB_CACHE', None),
    )
    
    queries = [
        "What is BGE M3?",
        "Defination of BM25"
    ] * 100
    passages = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"
    ] * 100
    
    queries_embeddings = model.encode_queries(
        queries,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    passages_embeddings = model.encode_corpus(
        passages,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    
    dense_scores = queries_embeddings["dense_vecs"] @ passages_embeddings["dense_vecs"].T
    sparse_scores = model.compute_lexical_matching_score(
        queries_embeddings["lexical_weights"],
        passages_embeddings["lexical_weights"],
    )

    print("Dense score:\n", dense_scores[:2, :2])
    print("Sparse score:\n", sparse_scores[:2, :2])


if __name__ == '__main__':
    test_m3_multi_devices()

    print("--------------------------------")
    print("Expected Output:")
    print("Dense score:")
    print(" [[0.626  0.3477]\n [0.3499 0.678 ]]")
    print("Sparse score:")
    print(" [[0.19561768 0.00878906]\n [0.         0.18030453]]")
