# echoChain: Information Degradation Analysis

## 1. Problem Statement
Build a small application that **measures how quickly technical information degrades** when it is passed through a “telephone‑game” chain of different large‑language‑model (LLM) APIs.  
The hypothesis is that repeatedly transforming an *answer* into a *question* and back into an *answer* will cause the core technical meaning to erode over successive cycles.

## 2. Telephone Process

| Phase | Prompt | Expected Output |
|-------|--------|-----------------|
| **A → Q** | “What question does this answer?” | A concise technical **question** for which the supplied answer is the direct response. |
| **Q → A** | “What’s the answer to this question?” | A concise, technically accurate **answer** to the supplied question. |

The system alternates these two phases until the content becomes meaningless or a maximum number of cycles is reached.

### Example Chain

| Cycle | Prompt | Model Response |
|-------|--------|----------------|
| 0 | *Seed answer* | `Redis uses single‑threaded event loops with epoll for I/O multiplexing.` |
| 1 | “What question does this answer?” | `How does Redis achieve high concurrency without multiple threads?` |
| 2 | “What’s the answer to this question?” | `Redis uses asynchronous I/O and an event‑driven architecture.` |
| 3 | “What question does this answer?” | `What techniques help Redis handle many connections efficiently?` |
| … | … | … |

## 3. Implementation Requirements

### 3.1 LLM Selection
* Connect to **≥ 3** models through **OpenRouter**.  
* **Recommended models**: `gemini‑2.0‑flash`, `claude‑4‑sonnet`, `gpt‑4.1‑mini`.  
  * Reserve **GPT‑4.1‑mini** exclusively as the **judge model**.

### 3.2 Prompt Templates

```text
# ANSWER → QUESTION
Given the following statement, formulate a concise technical question for which this
statement is the direct and most appropriate answer.
Do not be conversational. Output only the question.

Statement: "{answer_text}"
Question:
```

```text
# QUESTION → ANSWER
Provide a concise and technically accurate answer to the following question.
Do not be conversational. Output only the answer.

Question: "{question_text}"
Answer:
```

### 3.3 Rotation & Robustness
* Rotate through different LLM providers for each QA pair.  
* Gracefully handle API failures, rate limits, and token exhaustion.

### 3.4 Judging (LLM‑as‑a‑Judge)
* **Judge System Prompt**

```text
You are an expert technical evaluator. Your responses MUST be in JSON format.
```

* **Judge User Prompt**

```text
You will be given an Original Answer and a Transformed Answer.
Your task is to assess how much of the core technical meaning and information from the
Original Answer is preserved in the Transformed Answer.
Focus on factual accuracy and semantic similarity.

Original Answer:
{original_text}

Transformed Answer:
{transformed_text}

Provide your evaluation as a JSON object with a single key "score", where the value is an
integer between 1 (completely unrelated) and 10 (perfectly preserved).
Example:
{"score": 8}
```

* Call the judge **at least once per cycle** and average scores if multiple evaluations are taken.  
* **Degradation threshold**: stop when the average score ≤ 2 or after 30 cycles (whichever comes first).

### 3.5 Persistence
* Persist every QA pair and its judge score (e.g., JSONL or CSV) under an `experiments/` directory.  
* Store: cycle index, prompt type, text, model used, and score.

## 4. Command‑Line Example

```bash
$ python -m echochain --start "Redis uses single-threaded event loops with epoll for I/O multiplexing"
```

On completion the script prints the final status and writes results, e.g.:

```
DEGRADATION THRESHOLD REACHED (cycle 12) – results saved to experiments/redis_001/
```

## 5. Acceptance Criteria

1. **Deliverables**  
   * Zip file containing source code.  
   * `README.md` with setup and run instructions.

2. **Functional**  
   * Works out‑of‑the‑box after `pip install -e ".[dev]"`.  
   * Executes **≥ 3 experiments per test case** across **≥ 3 distinct test cases**.

3. **Reporting**  
   * Persisted results for each experiment.  
   * Brief write‑up covering:
     * Implementation approach.
     * Challenges & resolutions.
     * Major design decisions.

4. **Nice‑to‑Have**  
   * Summary statistics (average cycles to degradation).  
   * Hypothesis on LLM combinations that degrade information fastest/slowest.

## 6. Test Cases

Choose at least three of the following seeds:
1. `Redis uses single-threaded event loops with epoll for I/O multiplexing`
2. `TCP congestion control uses slow start and AIMD algorithms to prevent network overload`
3. `B-tree database indices maintain sorted data with O(log n) search complexity`
4. `Docker containers use copy-on-write filesystems for efficient storage layering`
5. `JWT authentication tokens contain base64-encoded JSON with HMAC signatures for security`
6. `Kubernetes uses etcd for distributed configuration storage with Raft consensus`

## 7. FAQ

| Question | Answer |
|----------|--------|
| **Can I use generative AI to help?** | Yes. Document how you used it (e.g., “Cursor autocomplete for type hints”). |
| **May I run more experiments than the minimum?** | Absolutely—note this in your write‑up. |
| **Do I have to use Python?** | Any language is allowed; justify the choice if not Python. |
