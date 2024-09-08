from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def get_spark() -> SparkSession:
    _builder = SparkSession.builder.config(  # type: ignore
        "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
    ).config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    spark: SparkSession = configure_spark_with_delta_pip(
        _builder,
        [
            "org.apache.hadoop:hadoop-azure:3.3.4",
            "io.delta:delta-sharing-spark_2.12:3.2.0",
        ],
    ).getOrCreate()
    return spark
