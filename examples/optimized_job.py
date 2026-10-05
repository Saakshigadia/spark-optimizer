"""The same report after applying the analyser's and optimizer's suggestions."""
from pyspark.sql import functions as F


def run(spark, orders, countries):
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.shuffle.partitions", "8")            # sized for the data, not the default 200

    enriched = (orders
                .join(F.broadcast(countries), "country_id")           # small table: broadcast, no shuffle
                .withColumn("total", F.col("amount") * (1 + F.col("tax_rate")))   # built-in, no Python UDF
                .cache())                                             # reused by three actions

    n_orders = enriched.count()
    revenue = enriched.groupBy("region").agg(F.sum("total").alias("revenue")).collect()
    top10 = enriched.orderBy(F.desc("total")).limit(10).collect()   # top-N instead of sorting everything to the driver
    enriched.unpersist()
    return n_orders, sorted((r["region"], round(r["revenue"], 2)) for r in revenue), [r["order_id"] for r in top10]
