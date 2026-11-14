from langchain_openai import OpenAIEmbeddings
import numpy as np


embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

def simple_embedding_demo():
    # single text
    text = "This is a sample text to be embedded."
    embedding = embeddings_model.embed_query(text)
    print(f"Embedding for single text: {embedding[:10]}\n") # will be 1536 values, so show first 10

    print(len(embedding))  # Should print 1536 for text-embedding-3-small


def basic_embeddings():

    # single text
    text = "What is Machine Learning?"
    single_embedding = embeddings_model.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")

    """
    numpy.linalg.norm()

    This prints the L2 norm (magnitude) of the embedding vector. Useful for:
    1. Sanity checking embeddings — embeddings should have reasonable magnitudes (not 0, not huge)
    2. Detecting normalization — if all norms are ≈ 1.0, the embeddings are L2-normalized (common in RAG/similarity search so that cosine similarity = dot product)
    3. Spotting degenerate vectors — near-zero norms indicate a bad embedding
    4. Debugging — outlier norms might indicate data issues

    In short,
    A norm = "size" of a vector/matrix. In your embedding code, np.linalg.norm(emb) gives the 
    Euclidean length of each embedding vector — a quick diagnostic for normalization, magnitude, 
    and degenerate outputs.

    """


def batch_embeddings():
    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]

    batch_embedding = embeddings_model.embed_documents(text)
    for i, emb in enumerate(batch_embedding):
        print(f"Text {i+1} - Vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")


def similarity_search():
    # Documents
    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    # embed documents and query
    doc_vector = embeddings_model.embed_documents(docs)
    query_vector = embeddings_model.embed_query(query)

    # compute cosine similarities
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f"  {score:.4f}: {doc}")


# # Caching ---
# def embedding_caching():
    from langchain_classic.embeddings.cache import CacheBackedEmbeddings

    from langchain_classic.storage import LocalFileStore
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        store = LocalFileStore(root_path=tempdir)

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings_model,
            document_embedding_cache=store,
            namespace="exercise",
        )

        text = "What is Reinforcement Learning?"

        # First call - hits API
        print("First call (API):")
        vectors1 = cached_embeddings.embed_documents([text])
        print(f"  Embedded {len(vectors1)} documents")

        # Second call - from cache
        print("\nSecond call (Cache):")
        vectors2 = cached_embeddings.embed_documents([text])
        print(f"  Embedded {len(vectors2)} documents")

        # Verify same results
        print(f"\nSame vectors: {np.allclose(vectors1[0], vectors2[0])}")


if __name__ == '__main__':
    # simple_embedding_demo()
    # basic_embeddings()
    # batch_embeddings()
    similarity_search()
    # embedding_caching()