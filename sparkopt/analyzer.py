"""Static code analyser for PySpark scripts and Spark SQL.

PySpark code is parsed with Python's `ast` module, so rules match real calls,
not text that happens to look similar (comments, strings). SQL is checked with
regular expressions after comments and string literals are removed.
"""
import ast
import re
from dataclasses import asdict, dataclass

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
ACTIONS = {"count", "collect", "show", "take", "first", "toPandas", "head", "foreach"}


@dataclass
class Finding:
    rule: str
    severity: str
    line: int
    title: str
    why: str
    fix: str

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- PySpark

class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.findings: list[Finding] = []
        self.loop_depth = 0
        self.actions_on: dict[str, list[int]] = {}
        self.cached: set[str] = set()

    def chain(self, node):
        """Method names called before this one, e.g. df.groupBy().agg().collect() -> {'groupBy', 'agg'}."""
        names, cur = set(), node.func.value if isinstance(node.func, ast.Attribute) else None
        while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
            names.add(cur.func.attr)
            cur = cur.func.value
        return names

    def visit_Module(self, node):
        # remember sorts that are immediately limited: df.orderBy(...).limit(n)
        self.limited = {id(n.func.value) for n in ast.walk(node)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr in {"limit", "take", "head"} and isinstance(n.func.value, ast.Call)}
        self.generic_visit(node)

    def add(self, *args):
        self.findings.append(Finding(*args))

    # loops -------------------------------------------------------------
    def visit_For(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    visit_While = visit_For

    # decorators: @udf / @F.udf ------------------------------------------
    def visit_FunctionDef(self, node):
        for d in node.decorator_list:
            target = d.func if isinstance(d, ast.Call) else d
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            if name == "udf":
                self.add("python-udf", "high", node.lineno, f"Python UDF `{node.name}`",
                         "Python UDFs move every row between the JVM and Python and block Catalyst optimisations.",
                         "Use built-in functions from pyspark.sql.functions, or a pandas_udf if no built-in exists.")
        self.generic_visit(node)

    def visit_Assign(self, node):
        # df = something.cache() / .persist()
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr in {"cache", "persist"}:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.cached.add(t.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        line = node.lineno

        if name == "udf":
            # udf(lambda x: ..., "double") or udf(my_func): flag it. udf("double") used as a decorator
            # has only a type string and is reported once on the function definition instead.
            if node.args and not isinstance(node.args[0], ast.Constant):
                self.add("python-udf", "high", line, "Python UDF",
                         "Python UDFs move every row between the JVM and Python and block Catalyst optimisations.",
                         "Use built-in functions from pyspark.sql.functions, or a pandas_udf if no built-in exists.")
        elif name == "collect" and not self.chain(node) & {"limit", "agg", "count", "distinct"}:
            self.add("collect", "high", line, "collect() pulls all rows to the driver",
                     "On large data this runs the driver out of memory and is slow over the network.",
                     "Keep the work distributed; use .limit(n).collect(), .take(n) or write the result out.")
        elif name == "toPandas":
            self.add("to-pandas", "high", line, "toPandas() on a Spark DataFrame",
                     "Converts the whole dataset on the driver; fails or crawls on large data.",
                     "Aggregate or .limit() first, and enable spark.sql.execution.arrow.pyspark.enabled.")
        elif name == "crossJoin":
            self.add("cross-join", "high", line, "Cartesian crossJoin",
                     "Output rows = rows(left) x rows(right); usually explodes in size.",
                     "Join on a key. If a cross join is really needed, make sure one side is tiny.")
        elif name == "groupByKey":
            self.add("group-by-key", "high", line, "RDD groupByKey",
                     "Ships every value across the network before combining, causing heavy shuffles and memory pressure.",
                     "Use reduceByKey/aggregateByKey, or the DataFrame groupBy().agg() API.")
        elif name in {"repartition", "coalesce"} and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 1:
            self.add("single-partition", "medium", line, f"{name}(1) forces one partition",
                     "All data goes through a single task on one executor, removing parallelism.",
                     "Write with several partitions; if one output file is required, coalesce only at the very end.")
        elif name == "join" and isinstance(f, ast.Attribute) and len(node.args) == 1 and not node.keywords:
            self.add("join-no-key", "high", line, "join() without a join condition",
                     "With no key Spark performs a cartesian product.",
                     "Pass the join column(s): df.join(other, on='id', how='inner').")
        elif name == "join" and isinstance(f, ast.Attribute) and not any(
                isinstance(a, ast.Call) and getattr(a.func, "attr", getattr(a.func, "id", "")) == "broadcast" for a in node.args):
            self.add("broadcast-hint", "low", line, "Check whether this join can be broadcast",
                     "Joining a large table to a small one with a shuffle join moves the large table over the network.",
                     "If one side is small (< ~100 MB), wrap it: df.join(F.broadcast(small_df), 'key').")
        elif name in {"orderBy", "sort"} and id(node) not in self.limited:
            self.add("global-sort", "medium", line, f"Global {name}()",
                     "A full sort shuffles every row into range partitions.",
                     "If you only need the top rows, use .orderBy(...).limit(n); sort within partitions if global order is not needed.")
        elif name == "withColumn" and self.loop_depth:
            self.add("with-column-loop", "medium", line, "withColumn() inside a loop",
                     "Each call adds a projection to the plan; long chains make planning very slow.",
                     "Build the columns in a list and add them with a single .select() or .withColumns({...}).")
        elif name in {"count", "collect", "show"} and self.loop_depth:
            self.add("action-in-loop", "high", line, f"{name}() inside a loop",
                     "Every iteration triggers a full Spark job.",
                     "Compute once outside the loop, or restructure as a single aggregation.")
        elif name == "set" and len(node.args) == 2 and all(isinstance(a, ast.Constant) for a in node.args):
            key, val = str(node.args[0].value), str(node.args[1].value).lower()
            if key == "spark.sql.adaptive.enabled" and val == "false":
                self.add("aqe-off", "medium", line, "Adaptive Query Execution turned off",
                         "AQE merges tiny shuffle partitions and fixes skewed joins at runtime.",
                         "Leave spark.sql.adaptive.enabled=true (the default since Spark 3.2).")
            elif key == "spark.sql.autoBroadcastJoinThreshold" and val == "-1":
                self.add("broadcast-off", "low", line, "Automatic broadcast joins disabled",
                         "Small tables will be shuffled instead of broadcast.",
                         "Remove this setting or raise the threshold (e.g. 64m).")
            elif key == "spark.sql.shuffle.partitions" and val == "200":
                self.add("default-partitions", "low", line, "Shuffle partitions left at the default 200",
                         "200 is too many for small data and too few for very large data.",
                         "Size it from the shuffle volume (about 128 MB per partition); see the optimizer.")
        elif name == "csv" and any(k.arg == "inferSchema" and isinstance(k.value, ast.Constant) and k.value.value for k in node.keywords):
            self.add("infer-schema", "low", line, "inferSchema=True",
                     "Spark reads the data an extra time to guess column types.",
                     "Pass an explicit schema, or store the data as Parquet which carries its schema.")

        # track actions per DataFrame variable to spot recomputation without cache
        if name in ACTIONS and isinstance(f, ast.Attribute):
            base = f.value
            while isinstance(base, ast.Call) and isinstance(base.func, ast.Attribute):
                base = base.func.value
            if isinstance(base, ast.Name):
                self.actions_on.setdefault(base.id, []).append(line)
        self.generic_visit(node)


def analyze_python(code: str) -> list[Finding]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Finding("syntax", "high", e.lineno or 1, "Code does not parse", str(e.msg), "Fix the syntax error first.")]
    v = _Visitor()
    v.visit(tree)
    for var, lines in v.actions_on.items():
        if len(lines) >= 2 and var not in v.cached and var not in {"spark", "sc"}:
            v.add("no-cache", "medium", lines[1], f"`{var}` is computed {len(lines)} times",
                  "Each action re-runs the whole lineage from the source unless the DataFrame is cached.",
                  f"Call {var} = {var}.cache() before the first action, and unpersist() when done.")
    return _sorted(v.findings)


# ---------------------------------------------------------------- SQL

def _strip_sql(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), sql, flags=re.S)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


SQL_RULES = [
    ("select-star", "medium", r"\bselect\s+\*", "SELECT *",
     "Reads every column; with Parquet/ORC you lose column pruning.", "List only the columns you need."),
    ("sql-cross-join", "high", r"\bcross\s+join\b", "CROSS JOIN",
     "Produces every combination of rows.", "Join on a key, or make sure one side is tiny."),
    ("implicit-join", "high", r"\bfrom\s+\w+(?:\s+(?!where|join|group|order|limit|left|right|inner|full|on)\w+)?\s*,\s*\w+", "Comma join (FROM a, b)",
     "Easy to forget the join condition, which turns it into a cross join.", "Use explicit JOIN ... ON."),
    ("not-in-subquery", "medium", r"\bnot\s+in\s*\(\s*select\b", "NOT IN (subquery)",
     "Handles NULLs surprisingly and is often planned as a slow nested-loop join.", "Use NOT EXISTS or a LEFT ANTI JOIN."),
    ("union-distinct", "low", r"\bunion\b(?!\s+all)", "UNION instead of UNION ALL",
     "UNION removes duplicates, which adds a shuffle.", "Use UNION ALL if duplicates are impossible or acceptable."),
    ("function-on-filter", "medium", r"\bwhere\b[^;]*?\b(year|month|date|to_date|substr|substring|cast|lower|upper)\s*\(", "Function on a filtered column",
     "Wrapping the column in a function can stop partition pruning and predicate pushdown.",
     "Filter on the raw column with a range, e.g. dt >= '2026-01-01' AND dt < '2027-01-01'."),
    ("leading-wildcard", "low", r"\blike\s+'%", "LIKE with a leading %",
     "Cannot use data skipping or min/max statistics.", "Avoid leading wildcards where possible."),
    ("count-distinct", "low", r"\bcount\s*\(\s*distinct\b", "COUNT(DISTINCT ...)",
     "Exact distinct counts need a large shuffle.", "If an estimate is fine, use approx_count_distinct()."),
]


def analyze_sql(sql: str) -> list[Finding]:
    clean = _strip_sql(sql)
    low = clean.lower()
    findings = []
    for rule, sev, pattern, title, why, fix in SQL_RULES:
        for m in re.finditer(pattern, low, flags=re.S):
            findings.append(Finding(rule, sev, low.count("\n", 0, m.start()) + 1, title, why, fix))
    # ORDER BY at the end of a statement without LIMIT
    for stmt_start, stmt in _statements(low):
        m = re.search(r"\border\s+by\b", stmt)
        if m and not re.search(r"\blimit\s+\d+", stmt) and not re.search(r"\bover\s*\(", stmt):
            pos = stmt_start + m.start()
            findings.append(Finding("order-no-limit", "medium", low.count("\n", 0, pos) + 1, "ORDER BY without LIMIT",
                                    "Sorting the full result is a global shuffle.", "Add LIMIT, or drop the ORDER BY if order does not matter."))
    return _sorted(findings)


def _statements(sql):
    pos = 0
    for part in sql.split(";"):
        if part.strip():
            yield pos, part
        pos += len(part) + 1


def _sorted(findings):
    return sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.line))


def analyze(code: str, language: str | None = None) -> list[Finding]:
    """Auto-detects SQL vs PySpark if language is not given."""
    if language is None:
        language = "sql" if re.match(r"\s*(--.*\n\s*)*(select|with|insert|create)\b", code, re.I) else "python"
    return analyze_sql(code) if language == "sql" else analyze_python(code)


def score(findings) -> int:
    """0-100 health score: 100 means no issues found."""
    penalty = sum({"high": 15, "medium": 7, "low": 2}[f.severity] for f in findings)
    return max(0, 100 - penalty)
