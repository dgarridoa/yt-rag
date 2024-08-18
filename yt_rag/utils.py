import os
from typing import Iterable

import mlflow
from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType


def create_delta_table(
    spark: SparkSession,
    schema: StructType,
    table_uri: str,
    partition_cols: list[str] | None = None,
    path: str | None = None,
) -> None:
    delta_table_builder = (
        DeltaTable.createIfNotExists(spark)
        .tableName(table_uri)
        .addColumns(schema)
    )

    if partition_cols:
        delta_table_builder = delta_table_builder.partitionedBy(partition_cols)

    if path:
        delta_table_builder.location(path).execute()
    else:
        delta_table_builder.execute()


def write_delta_table(
    spark: SparkSession,
    df: DataFrame,
    schema: StructType,
    table_uri: str,
    mode: str = "overwrite",
    partition_cols: list[str] | None = None,
    path: str | None = None,
) -> None:
    create_delta_table(spark, schema, table_uri, partition_cols, path)
    df = df.select(*schema.fieldNames())

    data_frame_writter = df.write.format("delta").mode(mode)

    if partition_cols:
        data_frame_writter = data_frame_writter.partitionBy(
            *partition_cols
        ).option("partitionOverwriteMode", "dynamic")

    data_frame_writter.saveAsTable(table_uri)


def set_mlflow_experiment() -> None:
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not experiment_name:
        raise ValueError(
            "environment variable MLFLOW_EXPERIMENT_NAME is unset"
        )
    mlflow.set_experiment(experiment_name)


def remove_columns_from_schema(
    schema: StructType, columns: Iterable[str]
) -> StructType:
    return StructType([field for field in schema if field.name not in columns])


def get_table_info(spark: SparkSession, table_uri: str) -> dict[str, str]:
    table_info = spark.sql(f"DESCRIBE EXTENDED {table_uri}")
    table_info_dict = {
        row["col_name"]: row["data_type"] for row in table_info.collect()
    }
    return table_info_dict
