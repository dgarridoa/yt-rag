from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

VideoTranscriptSchema = StructType(
    [
        StructField(
            "video",
            StructType(
                [
                    StructField("channel_id", StringType()),
                    StructField("video_id", StringType()),
                    StructField("title", StringType()),
                    StructField("publish_time", TimestampType()),
                ],
            ),
        ),
        StructField(
            "transcript",
            ArrayType(
                StructType(
                    [
                        StructField("start", DoubleType()),
                        StructField("duration", DoubleType()),
                        StructField("text", StringType()),
                    ]
                )
            ),
        ),
    ]
)
TranscriptsSchema = StructType(
    [
        StructField("channel_id", StringType()),
        StructField("video_id", StringType()),
        StructField("title", StringType()),
        StructField("publish_time", TimestampType()),
        StructField("start", DoubleType()),
        StructField("duration", DoubleType()),
        StructField("text", StringType()),
    ]
)

ContentSchema = StructType(
    [
        StructField("channel_id", StringType()),
        StructField("video_id", StringType()),
        StructField("title", StringType()),
        StructField("publish_time", TimestampType()),
        StructField("content", StringType()),
    ]
)

ContentChunksSchema = StructType(
    [
        StructField("id", StringType()),
        StructField("channel_id", StringType()),
        StructField("video_id", StringType()),
        StructField("title", StringType()),
        StructField("publish_time", TimestampType()),
        StructField("content", StringType()),
    ]
)
