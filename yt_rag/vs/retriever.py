import faiss
import numpy as np
import pandas as pd
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from yt_rag.params import VectorStoreRetrieverParams


def create_hnsw_index(
    df: pd.DataFrame, M: int = 32, efConstruction: int = 40, efSearch: int = 16
):
    embeddings = np.array(df["embedding"].tolist())
    d = embeddings.shape[1]
    M = 32
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    index.add(embeddings)  # type: ignore
    return index


def get_docs(df: pd.DataFrame):
    docs = {}
    for _, row in df.iterrows():
        docs[row["id"]] = Document(
            page_content=row["content"],  # type: ignore
            metadata={
                "id": row["id"],
                "video_id": row["video_id"],
                "title": row["title"],
                "publish_time": row["publish_time"],
            },
        )
    return docs


def get_index_to_docstore_id(df):
    return df["id"].to_dict()


def create_vector_store_retriever(
    params: VectorStoreRetrieverParams,
    df: pd.DataFrame,
    embedding_function: Embeddings,
) -> VectorStoreRetriever:
    index = create_hnsw_index(
        df, params.M, params.efConstruction, params.efSearch
    )
    docs = get_docs(df)
    index_to_docstore_id = get_index_to_docstore_id(df)
    docstore = InMemoryDocstore(docs)
    retriever = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    ).as_retriever(search_kwargs={"k": params.k})
    return retriever
