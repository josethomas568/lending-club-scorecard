from pyspark.sql import SparkSession


def spark_session(cfg, app: str) -> SparkSession:
    spark = (SparkSession.builder.appName(app)
             .master(cfg["spark"]["master"])
             .config("spark.driver.memory", cfg["spark"]["driver_memory"])
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", "16")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark
