from datetime import datetime, timezone

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from yt_rag.params import VectorStoreRetrieverParams
from yt_rag.vs.retriever import create_vector_store_retriever


def test_create_vector_store_retriever():
    conf = {
        "env": "default",
        "database": "default",
        "table_url": "config.share#yt-rag-embeddings.dev.content_chunks_embeddings",
        "M": 32,
        "efConstruction": 40,
        "efSearch": 10,
        "k": 2,
    }
    params = VectorStoreRetrieverParams.model_validate(conf)
    embedding_function = DeterministicFakeEmbedding(size=2)
    df = pd.DataFrame(
        {
            "id": [
                f"UC34rhn8Um7R18-BHjPklYlw_Guy5D3PJlZk_{i}" for i in range(4)
            ],
            "channel_id": "UC34rhn8Um7R18-BHjPklYlw",
            "video_id": "Guy5D3PJlZk",
            "title": "Agile Manifesto",
            "publish_time": datetime(
                2024, 8, 9, 16, 3, 23, tzinfo=timezone.utc
            ),
            "content": [
                "I often make this joke which is agile's a lot like communism you know people just keep not trying it correctly um",
                "not trying it correctly um what is what is the correct way to Agile oh gee um it's a real simple idea",
                "a real simple idea right uh do things in really short sequences measure how much you get done in every sequence use that",
                "in every sequence use that measurement to project an end date and tell everybody that's kind of it",
            ],
            "embedding": [
                [-0.34710384916349935, -1.1611512293332775],
                [-1.708373551760354, 0.33429172440673527],
                [0.07570851080112607, -0.3293924052367194],
                [0.5019245986862018, -0.2722309603366409],
            ],
        }
    )
    retriever = create_vector_store_retriever(params, df, embedding_function)
    docs = retriever.invoke(
        input="I often make this joke which is agile's a lot like communism you know people just keep not trying it correctly um"
    )
    expected_docs = [
        Document(
            metadata={
                "id": "UC34rhn8Um7R18-BHjPklYlw_Guy5D3PJlZk_0",
                "video_id": "Guy5D3PJlZk",
                "title": "Agile Manifesto",
                "publish_time": pd.Timestamp(
                    "2024-08-09 16:03:23+0000", tz="UTC"
                ),
            },
            page_content="I often make this joke which is agile's a lot like communism you know people just keep not trying it correctly um",
        ),
        Document(
            metadata={
                "id": "UC34rhn8Um7R18-BHjPklYlw_Guy5D3PJlZk_2",
                "video_id": "Guy5D3PJlZk",
                "title": "Agile Manifesto",
                "publish_time": pd.Timestamp(
                    "2024-08-09 16:03:23+0000", tz="UTC"
                ),
            },
            page_content="a real simple idea right uh do things in really short sequences measure how much you get done in every sequence use that",
        ),
    ]
    assert docs == expected_docs
