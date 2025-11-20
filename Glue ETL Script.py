import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, to_date, upper, coalesce, lit
from awsglue.dynamicframe import DynamicFrame

## Initialize contexts
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# --- Define S3 Paths (Updated with your new names) ---
s3_input_path = "s3://vm-reviews-landing/"
s3_processed_path = "s3://vm-reviews-processed/processed-data/"
s3_analytics_path = "s3://vm-reviews-processed/Athena Results/"

# --- Read the data from the S3 landing zone ---
dynamic_frame = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": [s3_input_path], "recurse": True},
    format="csv",
    format_options={"withHeader": True, "inferSchema": True},
)

# Convert to a standard Spark DataFrame for easier transformation
df = dynamic_frame.toDF()

# --- Perform Transformations ---
# 1. Cast 'rating' to integer and fill null values with 0
df_transformed = df.withColumn("rating", coalesce(col("rating").cast("integer"), lit(0)))

# 2. Convert 'review_date' string to a proper date type
df_transformed = df_transformed.withColumn("review_date", to_date(col("review_date"), "yyyy-MM-dd"))

# 3. Fill null review_text with a default string
df_transformed = df_transformed.withColumn("review_text",
    coalesce(col("review_text"), lit("No review text")))

# 4. Convert product_id to uppercase for consistency
df_transformed = df_transformed.withColumn("product_id_upper", upper(col("product_id")))

# 5. Convert 'customer_id' to uppercase for consistency in analysis
df_transformed = df_transformed.withColumn("customer_id", upper(col("customer_id")))

# --- Write the full transformed data to S3 (Good practice) ---
# This saves the clean, complete dataset to the 'processed-data' folder
glue_processed_frame = DynamicFrame.fromDF(df_transformed, glueContext, "transformed_df")
glueContext.write_dynamic_frame.from_options(
    frame=glue_processed_frame,
    connection_type="s3",
    connection_options={"path": s3_processed_path},
    format="csv"
)

# --- Run Spark SQL Query within the Job ---

# 1. Create a temporary view in Spark's memory
df_transformed.createOrReplaceTempView("product_reviews")

# 2. Run your SQL query
df_analytics_result = spark.sql("""
    SELECT 
        product_id_upper, 
        AVG(rating) as average_rating,
        COUNT(*) as review_count
    FROM product_reviews
    GROUP BY product_id_upper
    ORDER BY average_rating DESC
""")

# 3. Write the query's result DataFrame to your 'Athena Results' path
print(f"Writing analytics results to {s3_analytics_path}...")

# repartition(1) writes the result as a single file
analytics_result_frame = DynamicFrame.fromDF(df_analytics_result.repartition(1), glueContext, "analytics_df")
glueContext.write_dynamic_frame.from_options(
    frame=analytics_result_frame,
    connection_type="s3",
    connection_options={"path": s3_analytics_path},
    format="csv"
)

# Write the spark queries for following:
# 2. Date wise review count: his query calculates the total number of reviews submitted per day.
# 3. Top 5 Most Active Customers: This query identifies your "power users" by finding the customers who have submitted the most reviews.
# 4. Overall Rating Distribution: This query shows the count for each star rating (1-star, 2-star, etc.)

# This query calculates the total number of reviews submitted per day.
print("Running Query 2: Daily Review Count...")
df_analytics_result_2 = spark.sql("""
    SELECT
        review_date,
        COUNT(*) as daily_review_count
    FROM product_reviews
    GROUP BY review_date
    ORDER BY review_date ASC
""")

# Write the query's result DataFrame to its dedicated path
analytics_path_2 = s3_analytics_path + "daily_counts/"
print(f"Writing Query 2 results to {analytics_path_2}...")
analytics_result_frame_2 = DynamicFrame.fromDF(df_analytics_result_2.repartition(1), glueContext, "analytics_df_2")
glueContext.write_dynamic_frame.from_options(
    frame=analytics_result_frame_2,
    connection_type="s3",
    connection_options={"path": analytics_path_2},
    format="csv"
)

# --- Query 3: Top 5 Most Active Customers ---
# This query identifies the top 5 customers by the number of reviews submitted.
print("Running Query 3: Top 5 Active Customers...")
df_analytics_result_3 = spark.sql("""
    SELECT
        customer_id,
        COUNT(*) as total_reviews
    FROM product_reviews
    GROUP BY customer_id
    ORDER BY total_reviews DESC
    LIMIT 5
""")

# Write the query's result DataFrame to its dedicated path
analytics_path_3 = s3_analytics_path + "top_customers/"
print(f"Writing Query 3 results to {analytics_path_3}...")
analytics_result_frame_3 = DynamicFrame.fromDF(df_analytics_result_3.repartition(1), glueContext, "analytics_df_3")
glueContext.write_dynamic_frame.from_options(
    frame=analytics_result_frame_3,
    connection_type="s3",
    connection_options={"path": analytics_path_3},
    format="csv"
)


# --- Query 4: Overall Rating Distribution ---
# This query shows the count for each star rating (1-star, 2-star, etc.).
print("Running Query 4: Overall Rating Distribution...")
df_analytics_result_4 = spark.sql("""
    SELECT
        rating,
        COUNT(*) as rating_count
    FROM product_reviews
    -- Only count valid ratings (1 through 5)
    WHERE rating >= 1 AND rating <= 5
    GROUP BY rating
    ORDER BY rating ASC
""")

# Write the query's result DataFrame to its dedicated path
analytics_path_4 = s3_analytics_path + "rating_distribution/"
print(f"Writing Query 4 results to {analytics_path_4}...")
analytics_result_frame_4 = DynamicFrame.fromDF(df_analytics_result_4.repartition(1), glueContext, "analytics_df_4")
glueContext.write_dynamic_frame.from_options(
    frame=analytics_result_frame_4,
    connection_type="s3",
    connection_options={"path": analytics_path_4},
    format="csv"
)

job.commit()