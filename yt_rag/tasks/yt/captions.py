import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from yt_rag.params import Params, VideoCaptionsParams, read_config
from yt_rag.schemas import (
    ContentSchema,
    TranscriptsSchema,
    VideoTranscriptSchema,
)
from yt_rag.utils import write_delta_table
from yt_rag.yt.captions import (
    VideoTranscript,
    get_transcript_from_video,
    get_videos_from_channel,
)


class VideoCaptionsTask:
    def __init__(self, params: VideoCaptionsParams):
        self.params = params

    def get_video_transcripts(self, spark: SparkSession) -> DataFrame:
        videos = get_videos_from_channel(self.params.channel_id)
        video_transcripts = []
        for video in videos:
            transcript = get_transcript_from_video(video.video_id)
            video_transcripts.append(
                VideoTranscript(video=video, transcript=transcript)
            )
        df_video_transcripts = spark.createDataFrame(
            [vt.model_dump() for vt in video_transcripts],
            schema=VideoTranscriptSchema,
        )
        return df_video_transcripts

    def get_transcripts(self, df_video_transcripts: DataFrame) -> DataFrame:
        df_transcripts = df_video_transcripts.select(
            "video.channel_id",
            "video.video_id",
            "video.title",
            F.to_timestamp("video.publish_time").alias("publish_time"),
            F.explode("transcript").alias("transcript"),
        ).select(
            "channel_id",
            "video_id",
            "title",
            "publish_time",
            "transcript.start",
            "transcript.duration",
            "transcript.text",
        )
        return df_transcripts

    def get_content(self, df_video_transcripts: DataFrame) -> DataFrame:
        df_content = df_video_transcripts.select(
            "video.channel_id",
            "video.video_id",
            "video.title",
            F.to_timestamp("video.publish_time").alias("publish_time"),
            F.concat_ws(" ", F.col("transcript.text")).alias("content"),
        ).select(
            "channel_id",
            "video_id",
            "title",
            "publish_time",
            "content",
        )
        return df_content

    def read(self, spark: SparkSession, table_name: str) -> DataFrame:
        table_uri = f"{self.params.database}.{table_name}"
        if self.params.catalog:
            table_uri = f"{self.params.catalog}.{table_uri}"
        return spark.read.table(table_uri)

    def write(
        self,
        spark: SparkSession,
        df: DataFrame,
        schema: StructType,
        table_name: str,
    ):
        table_uri = f"{self.params.database}.{table_name}"
        if self.params.catalog:
            table_uri = f"{self.params.catalog}.{table_uri}"
        write_delta_table(spark, df, schema, table_uri, "overwrite")

    def launch(self, spark: SparkSession):
        df_video_transcripts = self.get_video_transcripts(spark)
        df_transcripts = self.get_transcripts(df_video_transcripts)
        df_content = self.get_content(df_video_transcripts)
        self.write(spark, df_transcripts, TranscriptsSchema, "transcripts")
        self.write(spark, df_content, ContentSchema, "content")


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore
    task = VideoCaptionsTask(params.video_captions)
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
