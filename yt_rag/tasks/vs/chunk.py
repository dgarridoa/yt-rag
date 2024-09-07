from typing import Iterator

import pandas as pd
import pyspark.sql.functions as F
from llama_index.core.node_parser import (
    TokenTextSplitter,
)
from llama_index.core.schema import Document
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, StringType, StructType

from yt_rag.params import ChunkParams, Params, read_config
from yt_rag.schemas import ContentChunksSchema
from yt_rag.utils import write_delta_table


def read_as_chunk(
    splitter: TokenTextSplitter, batch_iter: Iterator[pd.Series]
) -> Iterator[pd.Series]:
    def extract_and_split(text: str) -> list[str]:
        nodes = splitter.get_nodes_from_documents([Document(text=text)])
        return [n.get_content() for n in nodes]

    for x in batch_iter:
        yield pd.Series([extract_and_split(text) for text in x])


class ChunkTask:
    def __init__(self, params: ChunkParams):
        self.params = params

    def read(self, spark: SparkSession, table_name: str) -> DataFrame:
        table_uri = f"{self.params.database}.{table_name}"
        if self.params.catalog:
            table_uri = f"`{self.params.catalog}`.{table_uri}"
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
            table_uri = f"`{self.params.catalog}`.{table_uri}"
        write_delta_table(spark, df, schema, table_uri, "overwrite")

    def chunk(self, df_content: DataFrame):
        splitter = TokenTextSplitter(
            chunk_size=self.params.chunk_size,
            chunk_overlap=self.params.chunk_overlap,
        )

        @F.pandas_udf(ArrayType(StringType()))  # type: ignore
        def read_as_chunk_udf(
            batch_iter: Iterator[pd.Series],
        ) -> Iterator[pd.Series]:
            return read_as_chunk(splitter, batch_iter)

        columns = [col for col in df_content.columns if col != "content"]
        df_content_chunks = (
            df_content.select(
                *columns,
                F.posexplode(read_as_chunk_udf("content")).alias(  # type: ignore
                    "index", "content"
                ),
            )
            .withColumn(
                "id",
                F.concat(
                    "channel_id", F.lit("_"), "video_id", F.lit("_"), "index"
                ),
            )
            .drop("index")
        )
        return df_content_chunks

    def launch(self, spark: SparkSession):
        df_content = self.read(spark, "content")
        df_content_chunks = self.chunk(df_content)
        self.write(
            spark,
            df_content_chunks,
            ContentChunksSchema,
            "content_chunks",
        )


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore
    task = ChunkTask(params.chunk)
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
