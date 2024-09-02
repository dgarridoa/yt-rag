from datetime import datetime, timezone

import pandas as pd
import pytest
from pyspark.sql import Row, SparkSession

from tests.utils import assert_pyspark_df_equal
from yt_rag.params import ChunkParams
from yt_rag.schemas import ContentChunksSchema, ContentSchema
from yt_rag.tasks.vs.chunk import ChunkTask
from yt_rag.utils import write_delta_table

conf = {
    "env": "default",
    "database": "default",
    "chunk_size": 25,
    "chunk_overlap": 5,
}
params = ChunkParams.model_validate(conf)


def create_content_table(spark: SparkSession):
    df = spark.createDataFrame(
        [
            Row(
                channel_id="UC34rhn8Um7R18-BHjPklYlw",
                video_id="Guy5D3PJlZk",
                title="Agile Manifesto",
                publish_time=datetime(
                    2024, 8, 9, 16, 3, 23, tzinfo=timezone.utc
                ),
                content=" ".join(
                    [
                        "I often make this joke which is agile's",
                        "a lot like communism you know people",
                        "just keep not trying it correctly um",
                        "what is what is the correct way to",
                        "Agile oh gee um it's a real simple idea",
                        "right uh do things in really short",
                        "sequences measure how much you get done",
                        "in every sequence use that measurement",
                        "to project an end date and tell",
                        "everybody that's kind of it",
                    ]
                ),
            )
        ],
        schema=ContentSchema,
    )
    table_uri = f"{params.database}.content"
    write_delta_table(spark, df, ContentSchema, table_uri)


@pytest.fixture(scope="module")
def task(spark: SparkSession):
    create_content_table(spark)
    task = ChunkTask(params)
    task.launch(spark)
    return task


def test_content_chunks(spark: SparkSession, task: ChunkTask):
    df_content_chunks = task.read(spark, "content_chunks")
    df_expected_content_chunks = spark.createDataFrame(
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
    assert_pyspark_df_equal(df_expected_content_chunks, df_content_chunks)
