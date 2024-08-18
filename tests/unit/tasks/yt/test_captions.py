from datetime import datetime, timezone

import pandas as pd
import pytest
from pyspark.sql import Row, SparkSession

from tests.utils import assert_pyspark_df_equal
from yt_rag.params import VideoCaptionsParams
from yt_rag.schemas import ContentSchema, TranscriptsSchema
from yt_rag.tasks.yt.captions import VideoCaptionsTask

conf = {
    "env": "default",
    "database": "default",
    "channel_id": "UC34rhn8Um7R18-BHjPklYlw",
}
params = VideoCaptionsParams.model_validate(conf)


@pytest.fixture(scope="module")
def task(spark: SparkSession):
    task = VideoCaptionsTask(params)
    task.launch(spark)
    return task


def test_transcript(spark: SparkSession, task: VideoCaptionsTask):
    df_transcripts = task.read(spark, "transcripts")
    df_expected_transcripts = spark.createDataFrame(
        pd.DataFrame(
            {
                "channel_id": "UC34rhn8Um7R18-BHjPklYlw",
                "video_id": "Guy5D3PJlZk",
                "title": "Agile Manifesto",
                "publish_time": datetime(
                    2024, 8, 9, 16, 3, 23, tzinfo=timezone.utc
                ),
                "start": [
                    0.199,
                    2.32,
                    4.16,
                    7.12,
                    10.28,
                    15.2,
                    17.8,
                    19.92,
                    21.96,
                    23.92,
                ],
                "duration": [
                    3.961,
                    4.8,
                    6.12,
                    8.08,
                    7.52,
                    4.72,
                    4.16,
                    4.0,
                    6.04,
                    4.08,
                ],
                "text": [
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
                ],
            }
        ),
        schema=TranscriptsSchema,
    )
    assert_pyspark_df_equal(df_transcripts, df_expected_transcripts)


def test_content(spark: SparkSession, task: VideoCaptionsTask):
    df_content = task.read(spark, "content")
    df_expected_content = spark.createDataFrame(
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
    assert_pyspark_df_equal(df_content, df_expected_content)
