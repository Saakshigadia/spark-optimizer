"""Resource optimizer: recommend executors, memory, cores and shuffle partitions.

The sizing rules are the widely used Spark-on-YARN guidelines:
  - about 5 cores per executor (more hurts HDFS throughput, fewer wastes memory)
  - leave 1 core and 1 GB per node for the OS and node manager
  - memory overhead = max(384 MB, 10% of executor memory)
  - one executor's worth of resources goes to the driver / application master
  - shuffle partitions sized so each holds roughly 128 MB

The runtime estimate is a simple throughput model. Treat it as a first guess and
calibrate the MB/s numbers with timings from your own jobs.
"""
import math
from dataclasses import dataclass

# Rough per-core processing rate (MB/s) and how much data each job type shuffles.
JOB_PROFILES = {
    "etl":         {"mb_per_s": 60, "shuffle_ratio": 0.3},
    "aggregation": {"mb_per_s": 45, "shuffle_ratio": 0.5},
    "join":        {"mb_per_s": 30, "shuffle_ratio": 1.0},
    "ml":          {"mb_per_s": 15, "shuffle_ratio": 0.4},
}
SHUFFLE_MB_PER_S = 40          # per-core shuffle write + read rate
TARGET_PARTITION_MB = 128


@dataclass
class Cluster:
    nodes: int
    cores_per_node: int
    memory_gb_per_node: int


@dataclass
class Workload:
    input_gb: float
    job_type: str = "etl"
    sla_minutes: float = 60
    shuffle_ratio: float | None = None


def size_executors(cluster: Cluster):
    usable_cores = max(cluster.cores_per_node - 1, 1)
    cores_per_exec = min(5, usable_cores)
    execs_per_node = max(usable_cores // cores_per_exec, 1)
    total_execs = max(cluster.nodes * execs_per_node - 1, 1)          # one slot for the driver / AM
    mem_per_slot = (cluster.memory_gb_per_node - 1) / execs_per_node
    overhead = max(0.384, 0.10 * mem_per_slot)
    exec_mem = max(int(mem_per_slot - overhead), 1)
    return {"num_executors": total_execs, "executor_cores": cores_per_exec,
            "executor_memory_gb": exec_mem, "memory_overhead_gb": round(overhead, 2),
            "total_cores": total_execs * cores_per_exec}


def estimate_minutes(input_gb, job_type, shuffle_gb, total_cores):
    p = JOB_PROFILES[job_type]
    compute_s = input_gb * 1024 / (total_cores * p["mb_per_s"])
    shuffle_s = shuffle_gb * 1024 / (total_cores * SHUFFLE_MB_PER_S)
    return (compute_s + shuffle_s) * 1.15 / 60       # 15% for scheduling, stragglers and GC


def recommend(cluster: Cluster, workload: Workload):
    if workload.job_type not in JOB_PROFILES:
        raise ValueError(f"job_type must be one of {sorted(JOB_PROFILES)}")
    ex = size_executors(cluster)
    ratio = workload.shuffle_ratio if workload.shuffle_ratio is not None else JOB_PROFILES[workload.job_type]["shuffle_ratio"]
    shuffle_gb = workload.input_gb * ratio
    partitions = max(math.ceil(shuffle_gb * 1024 / TARGET_PARTITION_MB), ex["total_cores"] * 2)
    partitions = math.ceil(partitions / ex["total_cores"]) * ex["total_cores"]   # full waves, no idle cores

    minutes = estimate_minutes(workload.input_gb, workload.job_type, shuffle_gb, ex["total_cores"])
    sla_met = minutes <= workload.sla_minutes
    nodes_needed = cluster.nodes
    if not sla_met:
        nodes_needed = math.ceil(cluster.nodes * minutes / workload.sla_minutes)

    conf = {
        "spark.executor.instances": ex["num_executors"],
        "spark.executor.cores": ex["executor_cores"],
        "spark.executor.memory": f'{ex["executor_memory_gb"]}g',
        "spark.executor.memoryOverhead": f'{int(ex["memory_overhead_gb"] * 1024)}m',
        "spark.sql.shuffle.partitions": partitions,
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    }
    notes = [
        f'{ex["executor_cores"]} cores per executor, with 1 core and 1 GB per node kept for the OS.',
        f'About {shuffle_gb:.0f} GB will be shuffled, so {partitions} partitions keeps each near {TARGET_PARTITION_MB} MB.',
        "Adaptive Query Execution merges small partitions and fixes skewed joins at runtime.",
    ]
    if workload.job_type == "join":
        conf["spark.sql.autoBroadcastJoinThreshold"] = "64m"
        notes.append("Tables under 64 MB are broadcast, which avoids shuffling the large side of the join.")
    if not sla_met:
        notes.append(f"Estimated {minutes:.0f} min misses the {workload.sla_minutes:.0f} min SLA. "
                     f"About {nodes_needed} nodes would meet it.")

    submit = " ".join(["spark-submit"] + [f"--conf {k}={v}" for k, v in conf.items()] + ["your_job.py"])
    return {"executors": ex, "conf": conf, "spark_submit": submit, "shuffle_gb": round(shuffle_gb, 1),
            "estimated_minutes": round(minutes, 1), "sla_minutes": workload.sla_minutes, "sla_met": sla_met,
            "nodes_needed": nodes_needed, "notes": notes}
