# Model Degradation Analysis Pipeline: System Design

*Massively parallel pipeline that examines **information loss** when text flows through *chains* or *pairs* of LLMs, surfaces real-time compatibility scores, and feeds routing engines with evidence-based model-pair recommendations.*

---

## 1  Requirements Snapshot

| Axis | Target |
|------|--------|
| **Concurrency** | ≥ 10 k degradation chains executing in parallel |
| **Daily Volume** | ≥ 2 M model-pair evaluations (≈ 25 RPS sustained) |
| **Freshness** | Insights < 5 min after chain completion |
| **Extensibility** | Plug-in custom metrics (`semantic-sim`, `toxicity Δ`, etc.) via Python entry-points |
| **Storage Horizon** | 180 days hot, 2 years cold |
| **API Latency** | p95 ≤ 120 ms for “compatibility lookup” |
| **Availability** | 99.9 % SLO |

---

## 2  End-to-End Architecture

```mermaid
flowchart TB
  subgraph Control Plane
    UI[Web UI / CLI] --> CFG[Experiment Registry etcd]
    CFG --> PLAN[Planner / Scheduler]
  end

  PLAN --> Q(Kafka topic:<br/>deg.jobs)

  subgraph Data Plane (k8s)
    Q --> WP[Worker Pods Rust and Py plugins]
    WP --> LLM[Model Gateways]
    WP --> JUDGE[Judge LLM Pool]
    WP -- part-r.jsonl --> BUCKET(S3/Parquet)
    WP -. metrics .-> PROM[Prom + Grafana]
  end

  BUCKET --> STREAM[ClickHouse StreamInsert]
  STREAM --> CH[ClickHouse Cluster]
  CH --> FEEDS[Real-time Feature API gRPC and REST]

  FEEDS --> ROUTER[Production Model Router]
````

---

## 3  Control-Plane Primitives

### 3.1 Experiment Spec (YAML)

```yaml
id: "pairwise_2025_q4"
seeds:
  - "Explain Kubernetes pod scheduling."
  - "What is RSA encryption?"
models_src: ["gpt-4o","gemini-2-flash","claude-4-sonnet"]
models_tgt: ["mistral-medium","llama3-70b"]
metrics:
  - semantic_sim
  - factual_consistency
  - length_ratio
judge_model: "gpt-4o-mini"
hops: 6           # telephone depth; 1 => simple pairwise
priority: high
```

*Versioned in Git → pushed via GRPC to **Experiment Registry** (etcd key `/experiments/<id>/<semver>`).*

### 3.2 Planner / Scheduler

* Computes Cartesian set `(seed × src × tgt)`; splits into **job shards** = 500 chains.
* Publishes to **Kafka** (`deg.jobs`) with key `experiment_id::shard_id`.

---

## 4  Worker Pod Internals

| Stage             | Details                                                                                       |
| ----------------- | --------------------------------------------------------------------------------------------- |
| **Init**          | Loads Python metric plug-ins via entry-points (`pip install custom-metric-pkg`).              |
| **Execute Chain** | Alternates src → tgt models for `hops`; captures all intermediate answers.                    |
| **Judge**         | After each hop, calls **Judge LLM** with system prompt: *rate semantic drift 1-10*.           |
| **Partial Emit**  | Streams JSON Lines to **S3 multipart upload** every 5 s.                                      |
| **Retry**         | Exponential back-off (1 → 16 s) for transient 429/5xx; marks hop `error` field after 3 fails. |

*Pod template = Rust binary orchestrator + embedded Py interpreter (PyO3) for metrics; ensures 30–40 % lower CPU than pure Python.*

---

## 5  Metric & Storage Path

### 5.1 S3 → ClickHouse Ingest

*Parquet file naming:*
`s3://deg-data/experiment=<id>/date=YYYY-MM-DD/hour=HH/part-<uuid>.parquet`

**ClickHouse** `S3Cluster` pulls parts every 30 s; `INSERT SELECT` into hot table:

```sql
CREATE TABLE degradation_raw (
  expt String, src_model String, tgt_model String, hop UInt8,
  score Float32, custom_metrics Map(String, Float32),
  prompt String, response String, ts DateTime
) ENGINE = ReplicatedMergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (expt, src_model, tgt_model, hop, ts);
```

### 5.2 Roll-Up Views

* `compat_score_mv` — window=24 h average of final-hop scores → drives router.
* `drift_curve_mv` — hop × avg(score) for dashboards.
* `anomaly_mv` — Z-score on score Δ to flag sudden degradation.

---

## 6  Real-Time Feature API

* **gRPC** method: `GetCompatibility(src, tgt, constraint)` returns:

```json
{
  "score": 7.8,
  "conf_int": [7.6,8.0],
  "n": 1420,
  "trend_24h": -0.3,
  "recommended_hops": 3
}
```

*Reads from **ClickHouse Materialized View** cached in Redis for 30 s* → p95 ≈ 40 ms.

---

## 7  Custom Metric Plug-in Interface

```python
# my_metric/__init__.py
def setup():
    return {"name": "toxicity_delta", "version": "0.1.0"}


def compute(original: str, transformed: str) -> float:
    score_orig = toxicity_model.score(original)
    score_new = toxicity_model.score(transformed)
    return score_new - score_orig
```

*Workers search `pkg_resources.iter_entry_points('degradation_metrics')`.*

---

## 8  Live Recommendations for Router

* Router asks **Feature API** before choosing fallbacks.
* If `score < threshold` or `trend_24h < -Δ`, router downgrades `compatibility` feature weight.
* Ensures real-time avoidance of bad pairings.

---

## 9  Scaling & Reliability Measures

| Component      | Scaling                                     | Reliability                           |
| -------------- | ------------------------------------------- | ------------------------------------- |
| **Kafka**      | 64 partitions; MirrorMaker to DR region     | ISR=3; rack-aware                     |
| **Workers**    | Karpenter up to 400 pods; spot preferred    | PodDisruptionBudget minAvailable=80 % |
| **Judge Pool** | Stateless FastAPI; HPA on CPU & P95 latency | Circuit-breaker > 5 % 5xx             |
| **ClickHouse** | 3 shards × 3 replicas; tiered S3            | Async replication; keeper quorum      |

*Back-pressure:* Planner pauses new shards if ClickHouse lag > 5 min or Judge 5xx > 10 %.

---

## 10  Design Choices vs Alternatives

| Area                    | Picked                   | Rejected                          | Why                                 |
| ----------------------- | ------------------------ | --------------------------------- | ----------------------------------- |
| Storage staging         | Parquet on S3            | Direct DB writes                  | Isolates write bursts, cheap        |
| ClickHouse OLAP         | BigQuery                 | 0 \$ per-query, self-hosted SOC-2 |                                     |
| Worker Lang             | Rust+Py                  | Pure Go                           | Py plug-ins easy for ML metrics     |
| Judge LLM               | Single small GPT-4o-mini | Multiple heavy models             | Cost ×10 lower, still correlated    |
| Compatibility API cache | Redis                    | Memcached                         | Native TTL + LUA for conf\_int calc |

---

## 11  Likely Interview Follow-Ups

<details>
<summary>🔄 *How to add a new degradation metric without redeploying workers?*</summary>
Upload pip package to internal index; update Experiment YAML to reference metric; workers detect new entry-point on next shard spin-up (every 30 min) or via `SIGHUP` hot-reload.
</details>

<details>
<summary>📉 *What if Judge model drifts?*</summary>
Daily calibration suite; if median absolute error > 0.8 vs gold data, auto-swap to backup judge or retrain.
</details>

<details>
<summary>⚙️ *Can chains exceed 6 hops dynamically?*</summary>
Planner supports `adaptive_hops`: stop early if score slope Δ < ε for 2 steps to save tokens.
</details>

---

## 12  Conclusion

This pipeline combines **Kafka-based job fan-out**, **Rust workers with pluggable metrics**, **cheap Parquet staging**, and **ClickHouse real-time roll-ups** to deliver live degradation intelligence at scale—feeding Not Diamond’s router with fresh compatibility signals while remaining extensible, cost-efficient, and highly available.
