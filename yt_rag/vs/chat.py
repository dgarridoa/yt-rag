from langchain_community.chat_models.databricks import ChatDatabricks
from langchain_community.chat_models.mlflow import ChatMlflow
from langchain_core.messages import BaseMessage


def get_chat_model(
    endpoint_name: str = "databricks-meta-llama-3-1-405b-instruct",
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> ChatMlflow:
    chat_model = ChatDatabricks(
        endpoint=endpoint_name, temperature=temperature, max_tokens=max_tokens
    )
    return chat_model


def query_chat_model(chat_model: ChatMlflow, query: str) -> BaseMessage:
    return chat_model.invoke(query)
