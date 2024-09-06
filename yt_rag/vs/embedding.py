from langchain.embeddings import MlflowEmbeddings
from langchain_community.embeddings import DatabricksEmbeddings


def get_embedding_model(
    endpoint_name: str = "databricks-bge-large-en",
) -> MlflowEmbeddings:
    embedding_model = DatabricksEmbeddings(endpoint=endpoint_name)
    return embedding_model
