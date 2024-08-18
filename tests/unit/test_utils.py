import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession

from tests.utils import assert_pyspark_df_equal
from yt_rag.utils import (
    get_table_info,
    write_delta_table,
)


@pytest.fixture
def df(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        pd.DataFrame(
            {
                "channel_id": ["UC34rhn8Um7R18-BHjPklYlw"],
                "video_id": ["Guy5D3PJlZk"],
                "title": ["Agile Manifesto"],
                "content": ["I often make this joke which is agile's ..."],
            }
        ),
    )


def test_write_delta_table(spark: SparkSession, df: DataFrame):
    schema = df.schema
    database = "default"

    table_uri = f"{database}.internal_table"
    write_delta_table(spark, df, schema, table_uri)
    assert get_table_info(spark, table_uri)["Type"] == "MANAGED"
    delta_table_df = spark.read.table(table_uri)
    assert_pyspark_df_equal(df, delta_table_df)

    table_uri = f"{database}.internal_table_with_partitions"
    write_delta_table(
        spark, df, schema, table_uri, "overwrite", ["channel_id"]
    )
    assert get_table_info(spark, table_uri)["Type"] == "MANAGED"
    delta_table_df = spark.read.table(table_uri)
    assert_pyspark_df_equal(df, delta_table_df)
