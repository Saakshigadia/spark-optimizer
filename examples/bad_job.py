"""A typical first version of a sales report job. Works, but slow.
Run `python -m sparkopt.cli analyze examples/bad_job.py` to see what the analyser finds."""
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType


@udf(returnType=DoubleType())
def amount_with_tax(amount, rate):
    return float(amount) * (1 + float(rate))


def run(spark, orders, countries):
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")   # no automatic broadcast
    spark.conf.set("spark.sql.shuffle.partitions", "200")          # Spark default

    enriched = orders.join(countries, "country_id")
    enriched = enriched.withColumn("total", amount_with_tax("amount", "tax_rate"))

    n_orders = enriched.count()
    revenue = enriched.groupBy("region").agg(F.sum("total").alias("revenue")).collect()
    top10 = enriched.orderBy(F.desc("total")).collect()[:10]
    return n_orders, sorted((r["region"], round(r["revenue"], 2)) for r in revenue), [r["order_id"] for r in top10]
