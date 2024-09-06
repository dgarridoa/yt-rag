from unittest.mock import patch

with patch(
    "yt_rag.vs.embedding.get_embedding_model",
    lambda endpoint_name: DeterministicFakeEmbedding(size=2),
):
    from datetime import datetime, timezone

    import pandas as pd
    import pytest
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from pyspark.sql import SparkSession

    from tests.utils import assert_pyspark_df_equal
    from yt_rag.params import EmbeddingParams
    from yt_rag.schemas import (
        ContentChunksEmbeddingsSchema,
        ContentChunksSchema,
    )
    from yt_rag.tasks.vs.embedding import EmbeddingTask
    from yt_rag.utils import write_delta_table

    conf = {
        "env": "default",
        "database": "default",
        "endpoint_name": "databricks-bge-large-en",
    }
    params = EmbeddingParams.model_validate(conf)

    def create_content_chunks_table(spark: SparkSession):
        df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "id": [
                        f"UC34rhn8Um7R18-BHjPklYlw_Guy5D3PJlZk_{i}"
                        for i in range(4)
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
                }
            ),
            schema=ContentChunksSchema,
        )
        table_uri = f"{params.database}.content_chunks"
        write_delta_table(spark, df, ContentChunksSchema, table_uri)

    def get_dummy_embedding_model(endpoint_name: str = "dummy"):
        embed = DeterministicFakeEmbedding(size=2)
        return embed

    @pytest.fixture(scope="module")
    def task(spark: SparkSession):
        create_content_chunks_table(spark)
        task = EmbeddingTask(params)
        task.launch(spark)
        return task

    def test_content_chunks_embedding(
        spark: SparkSession, task: EmbeddingTask
    ):
        actual_df = task.read(spark, "content_chunks_embedding")
        expected_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "id": [
                        f"UC34rhn8Um7R18-BHjPklYlw_Guy5D3PJlZk_{i}"
                        for i in range(4)
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
            ),
            schema=ContentChunksEmbeddingsSchema,
        )
        assert_pyspark_df_equal(expected_df, actual_df)
