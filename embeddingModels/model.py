from langchain_community.embeddings import HuggingFaceEmbeddings

def load_selected_embeddings(model_name):
    valid_models = [
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "aditeyabaral/sentencetransformer-bert-base-cased",
        "WhereIsAI/UAE-Large-V1",
        "bert-large-uncased",
        "sembeddings/model_gpt_trained",
        "BAAI/bge-large-en-v1.5"
    ]

    if model_name not in valid_models:
        raise ValueError(f"Invalid model name: {model_name}")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
