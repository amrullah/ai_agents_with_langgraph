


from langchain_openai import OpenAIEmbeddings


embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

def simple_embedding_demo():
    # single text
    text = "This is a sample text to be embedded."
    embedding = embeddings.embed_query(text)
    print(f"Embedding for single text: {embedding[:10]}\n") # will be 1536 values, so show first 10

    print(len(embedding))  # Should print 1536 for text-embedding-3-small


if __name__ == '__main__':
    simple_embedding_demo()