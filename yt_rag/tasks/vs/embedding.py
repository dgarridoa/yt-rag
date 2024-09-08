import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, FloatType, StructType

from yt_rag.params import EmbeddingParams, Params, read_config
from yt_rag.schemas import ContentChunksEmbeddingsSchema
from yt_rag.utils import write_delta_table
from yt_rag.vs.embedding import get_embedding_model


class EmbeddingTask:
    def __init__(self, params: EmbeddingParams):
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

    def embedding(self, df_chunks: DataFrame):
        embedding_model = get_embedding_model(self.params.endpoint_name)

        @F.pandas_udf(ArrayType(FloatType()))  # type: ignore
        def get_embeddings_udf(contents: pd.Series) -> pd.Series:
            max_batch_size = 150
            batches = [
                list(contents.iloc[i : i + max_batch_size])
                for i in range(0, len(contents), max_batch_size)
            ]

            all_embeddings = []
            for batch in batches:
                try:
                    embedding = embedding_model.embed_documents(batch)
                except ValueError:
                    embedding = [None] * len(batch)
                all_embeddings += embedding

            return pd.Series(all_embeddings)

        df_chunks_embeddings = df_chunks.withColumn(
            "embedding",
            get_embeddings_udf("content"),  # type: ignore
        )
        return df_chunks_embeddings

    def launch(self, spark: SparkSession):
        df_chunks = self.read(spark, "content_chunks")
        df_chunks_embeddings = self.embedding(df_chunks)
        self.write(
            spark,
            df_chunks_embeddings,
            ContentChunksEmbeddingsSchema,
            "content_chunks_embedding",
        )


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore
    task = EmbeddingTask(params.embedding)
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
