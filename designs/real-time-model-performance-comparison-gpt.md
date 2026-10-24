# Real-Time Model Performance Comparison System

*A streaming analytics platform that ingests evaluation results from **hundreds of LLMs** over **thousands of tasks**, computes live leaderboards with confidence intervals and significance tests, and returns sub-second recommendations for the “best” model per task + metric.*

---

## 1  Constraints & Targets

| Axis | Requirement |
|------|-------------|
| **Volume** | ≥ 100 models × 5 000 tasks × 50 evals/day → **25 M rows/day** |
| **Freshness** | *p95* ≤ 60 s from eval finish → leaderboard |
| **Query Latency** | *p95* ≤ 120 ms for `/leaderboard?task=…` |
| **Custom Metrics** | User-pluggable Python functions (`bleu`, `bertscore`, custom) |
| **Stat Sig** | 95 % conf. intervals; z-test / bootstrap support |
| **High Availability** | 99.9 % API uptime |
| **Storage Horizon** | 365 days hot OLAP; cold S3 after 1 year |

---

## 2  Bird’s-Eye Architecture

```mermaid
flowchart TB
  subgraph Ingestion Plane
    WK[Eval Workers<br/>LLM benches] --> P1(Kafka topic:<br/>eval.raw)
  end

  P1 --> FLINK[Flink / Fluvio<br/>Stream Aggregator]
  FLINK --> FEATFeature KV<br/>Redis Cluster
  FLINK --> CKClickHouse Hot OLAP

  subgraph Query Plane
    API[Leaderboard & Rec API<br/>FastAPI + gRPC]
    API --> FEAT
    API --> CK
    API -->|cache| RDS[Redis Cache]
  end

  CK --> DASH[React + Grafana Dashboards]

  FLINK -. alerts .-> AM[Alertmanager ➜ PagerDuty]
````

---

## 3  Data Contracts

### 3.1 Raw Evaluation Message (Avro/JSON)

```jsonc
{
  "ts"          : 1725670123,
  "model_id"    : "gpt-4o",
  "task_id"     : "sst2_classification",
  "metric"      : "accuracy",
  "value"       : 0.934,
  "n_samples"   : 872,
  "eval_id"     : "uuid",
  "context"     : { "prompt_len":128, "temp":0.2 }
}
```

### 3.2 Aggregated Record (per model × task × metric × 1 min window)

```json
{
  "window_start" : "2025-10-07T12:30:00Z",
  "model_id"     : "gpt-4o",
  "task_id"      : "sst2_classification",
  "metric"       : "accuracy",
  "mean"         : 0.9312,
  "std_err"      : 0.0041,
  "n"            : 550,
  "wilson_low"   : 0.923,         // 95 % CI
  "wilson_high"  : 0.939
}
```

---

## 4  Stream Aggregator Logic (Flink Job)

1. **KeyBy** `(task_id, model_id, metric)`
2. **Sliding-Window** 1 min / 10 s hop; compute

   * mean, var, count
   * Wilson CI (for binomial metrics) or t-CI (others)
3. **Rank** per task/metric in real-time; store top-K (e.g., 10) in **Redis Feature KV** for instant API lookup.
4. **Emit** window aggregates to **ClickHouse** via `StreamInsert`.

---

## 5  OLAP Schema (ClickHouse)

```sql
CREATE TABLE eval_agg (
  window_start DateTime,
  task_id LowCardinality(String),
  model_id LowCardinality(String),
  metric  LowCardinality(String),
  mean    Float32,
  std_err Float32,
  wilson_low Float32,
  wilson_high Float32,
  n UInt32
) ENGINE = ReplicatedMergeTree
PARTITION BY toYYYYMMDD(window_start)
ORDER BY (task_id, metric, window_start, model_id);
```

Materialized views:

* **`leaderboard_latest_mv`** – latest row per `(task,metric,model)`.
* **`sig_test_mv`** – pair-wise z-test vs. current leader (`p_value`).

---

## 6  API Endpoints

| Route                                           | Purpose                         | Latency Path       |
| ----------------------------------------------- | ------------------------------- | ------------------ |
| `GET /v1/leaderboard?task=qa&metric=f1`         | Top-K models + CI               | Redis → 30 ms      |
| `GET /v1/compare?task=qa&m1=gpt-4o&m2=claude-4` | `z`-score, p-value              | ClickHouse → 80 ms |
| `POST /v1/metrics/plugin`                       | Upload custom metric wheel file | S3 ↔ CI pipeline   |

**Redis cache TTL 30 s**; cache key includes `task_id|metric`.

---

## 7  Custom Metric Plug-Ins

* Python wheel conforming to:

```python
def compute(predictions: List[str], references: List[str]) -> float: ...
```

* Uploaded via API → CI security scan → stored in **S3** → scheduled eval workers pull latest plugin manifest at start.

---

## 8  Confidence & Statistical Significance

* **Binomial (acc, exact match)** → *Wilson* CI.
* **Continuous (BLEU, F1)** → Student t-CI (assuming n > 30).
* **Pairwise Test**:

  ```python
  z = (μ1 − μ2) / sqrt(se1² + se2²)
  p = 1 − Φ(|z|)
  ```

*Flink writes `p_value` into `sig_test_mv`; dashboard highlights significant leaders.*

---

## 9  Scaling & Reliability

| Tier           | Scaling Strategy                                           |
| -------------- | ---------------------------------------------------------- |
| **Kafka**      | 128 partitions; ISR = 3; retention = 72 h                  |
| **Flink**      | Autoscale on lag; RocksDB state backend with S3 checkpoint |
| **ClickHouse** | 3 shards × 3 replicas → 500 k rows/s ingest                |
| **API**        | k8s HPA on RPS + p95 latency; Anycast ingress              |
| **Redis KV**   | Cluster mode; hash slot on `task_id`                       |

*Back-pressure* — if Flink lag > 5 min → Ingress returns 503 to eval workers to slow intake.

---

## 10  Key Design Decisions

| Topic          | Chosen               | Rejected                   | Justification                              |
| -------------- | -------------------- | -------------------------- | ------------------------------------------ |
| Stream engine  | **Flink**            | Spark Structured Streaming | Lower latency; exactly-once w/ Kafka       |
| Storage        | **ClickHouse**       | BigQuery / Druid           | Free queries, millisecond filter-on-prefix |
| Feature KV     | **Redis**            | DynamoDB                   | 1-2 ms lookups; in-VPC                     |
| CI calc        | Wilson / t-CI online | Bootstrap offline          | Fast, memory-light                         |
| Metric plug-in | Python wheels        | REST code snippets         | Easier sec-scan, version pinning           |

---

## 11  Likely Follow-Up Q\&A

<details>
<summary>🔄 *How to add a new task benchmark quickly?*</summary>
Upload dataset spec (`yaml + s3 uri`) via portal; evaluator scheduler picks up next cron; results appear automatically on leaderboard once ≥ 30 samples processed.
</details>

<details>
<summary>📉 *What if model count jumps to 1 000 ?*</summary>
Kafka partition scale; ClickHouse shards+1; Redis hash-slots auto-rebalance; Flink job parallelism increases (Karpenter adds nodes).
</details>

<details>
<summary>🔐 *Does raw prediction text ever leave the secure cluster?*</summary>
No; workers hash predictions before Kafka when `privacy=true`, and only metric scalar is published. For open benchmarks privacy=false to allow manual inspection.
</details>

---

## 12  Conclusion

By melding **Kafka + Flink** for streaming aggregates, **ClickHouse** for high-rate OLAP, and an in-memory **Redis** feature cache, the system delivers **near-real-time leaderboards** with statistical rigor while scaling to **hundreds of models** and **millions of evals per day**, empowering instant, confident model-selection decisions.
