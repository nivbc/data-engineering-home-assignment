import sys
from awsglue.transforms import *
from awsglue.dynamicframe import DynamicFrame
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, round, avg, stddev, sqrt, lit, date_sub, rank

# ------------------------------
# Initialization
# ------------------------------
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'output_path'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ------------------------------
# Load Data from s3
# ------------------------------
raw_s3_path = "s3://data-engineer-assignment-niv/rawData/"
dynamic_frame = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    format="csv",
    format_options={"withHeader": True},
    connection_options={"paths": [raw_s3_path]},
    transformation_ctx="raw_data"
)

df = dynamic_frame.toDF()

# ------------------------------
# Window Definitions
# ------------------------------
by_ticker_date = Window.partitionBy("ticker").orderBy("Date")

# ------------------------------
# Transformations
# ------------------------------
# Daily Returns
df = df.withColumn("prev_close", lag("close").over(by_ticker_date)) \
       .withColumn("daily_return", (col("close") - col("prev_close")) / col("prev_close"))

# Average daily return across all stocks
daily_returns_df = df.groupBy("Date") \
    .agg(round(avg("daily_return") * 100, 2).alias("average_return")) \
    .orderBy("Date")

# Average trading worth per stock
trading_worth_df = df.withColumn("trading_worth", col("close") * col("volume")) \
    .groupBy("ticker") \
    .agg(avg("trading_worth").alias("value")) \
    .orderBy(col("value").desc()) \
    .limit(1)

# Volatility (annualized stddev of daily return)
volatility_df = df.groupBy("ticker") \
    .agg(round(stddev("daily_return") * sqrt(lit(252)) * 100, 2).alias("standard_deviation")) \
    .orderBy(col("standard_deviation").desc()) \
    .limit(1)

# 30-day returns
historical_df = df.select("ticker", col("Date").alias("future_date"), col("close").alias("price_30_days_ago")) \
    .withColumn("Date", date_sub("future_date", 30)) \
    .drop("future_date")

df_with_returns = df.join(
    historical_df,
    on=["ticker", "Date"],
    how="left"
).withColumn(
    "return_30_day",
    round(((col("close") - col("price_30_days_ago")) / col("price_30_days_ago")) * 100, 2)
)

ticker_rank_window = Window.partitionBy("ticker").orderBy(col("return_30_day").desc())

top_returns_df = df_with_returns.select("ticker", "Date", "return_30_day") \
    .where(col("return_30_day").isNotNull()) \
    .withColumn("rank", rank().over(ticker_rank_window)) \
    .filter(col("rank") <= 3) \
    .orderBy(col("return_30_day").desc())
# ------------------------------
# Output Definitions - we can move it to yaml config file in next steps.
# ------------------------------
outputs = {
    "daily_returns": daily_returns_df,
    "trading_worth": trading_worth_df,
    "volatility": volatility_df,
    "top_returns": top_returns_df
}

# ------------------------------
# Save Outputs to S3 & Glue Catalog
# ------------------------------
for name, df in outputs.items():
    dyf = DynamicFrame.fromDF(df, glueContext, name)

    sink = glueContext.getSink(
        path=f"{args['output_path']}/{name}/",
        connection_type="s3",
        updateBehavior="UPDATE_IN_DATABASE",
        partitionKeys=[],
        compression="snappy",
        enableUpdateCatalog=True
    )

    sink.setCatalogInfo(
        catalogDatabase="vi_data_db",
        catalogTableName=name
    )

    sink.setFormat("glueparquet")
    sink.writeFrame(dyf)

# ------------------------------
# Commit Job
# ------------------------------
job.commit()
