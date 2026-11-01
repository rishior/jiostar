# Evaluation Report: Contract Intelligence Assistant

## 1. Executive Summary

We developed an automated legal clause retrieval and verification system evaluated on the LegalBench RAG CUAD benchmark (194 canonical queries across 29 commercial agreements). The assistant processes natural language inquiries regarding commercial contract covenants, rights, and restrictions, returning ranked passages with exact document paths and character offsets, or abstaining when an inquiry cannot be answered from the contract corpus.

Our v2 final architecture (`configs/final_v2.yaml`, derived from `iter_8_heading_aware_chunking`) combines:
1. Entity-aware document routing that matches contract and party entities in the query
2. Query prefix stripping (`strip_contract_prefix`) that isolates the substantive legal question from the agreement title, resolving prompt preamble bias
3. Structural heading-aware chunking (`extract_headings`) that extracts contractual article and section headers and prepends them to chunk embedding representations while preserving verbatim raw character coordinates in the source document
4. Fixed 512 character sliding windows with 128 character overlap aligned to word boundaries
5. Dense vector retrieval using normalized sentence embeddings (`all-MiniLM-L6-v2`) via FAISS inner product search
6. Calibrated score based abstention thresholding

On the full training partition (187 queries comprising 140 answerable questions and 47 unanswerable queries across both original and extra sets), the v2 final configuration achieved substantial gains across all primary metrics:
- Mean F1 at 5 increased from 0.0816 in v1 final to 0.1383 in v2 final (a 69.5 percent relative gain)
- Mean Precision at 5 increased from 0.0476 in v1 final to 0.0881 in v2 final (an 85.1 percent relative gain)
- Mean Recall at 5 increased from 0.4210 in v1 final to 0.4504 in v2 final
- Mean F1 at 1 surged from 0.0817 in v1 final to 0.1527 in v2 final (an 86.9 percent relative gain)
- Mean Precision at 1 jumped from 0.0658 in v1 final to 0.1571 in v2 final (a 138.8 percent relative gain)
- Mean Recall at 1 jumped from 0.1327 in v1 final to 0.1812 in v2 final
- Rank 1 offset 0 preamble rate plummeted from 40.7 percent down to 0.7 percent, demonstrating near total elimination of preamble bias
- In-corpus unanswerable abstention improved from 26.1 percent (6 of 23 rejected) in v1 to 47.8 percent (11 of 23 rejected) in v2
- Total unanswerable abstention reached 63.8 percent (30 of 47 rejected)
- False negative abstention on answerable queries remained at exactly 0.0 percent (zero false rejections)
- Query inference latency remained sub 30 milliseconds on standard CPU

Across project history, the test partition was evaluated 4 times in total. All 4 of those evaluations occurred before the v2 improvement loop, and none were used to select configurations in v2. During the entire v2 development and 6 iteration improvement loop, the test split was evaluated zero times. The test split is sequestered for the user to evaluate strictly once at the end using the command provided in Section 11.

## 2. System Architecture

The architecture prioritizes character-level fidelity, modular component isolation, and deterministic execution.

```
+---------------------------------------------------------------------------------------+
|                                    User Legal Query                                   |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                               Document Entity Router                                  |
|     Matches contract title, parties, and aliases against 29 indexed corpus agreements |
+---------------------------------------------------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
             [Contract Found]                              [Contract Missing]
                    |                                             |
                    v                                             v
+---------------------------------------+             +-----------------------+
|         Query Prefix Stripper         |             |       Abstention      |
|  Removes "Consider the Agreement...;" |             | (Out-of-corpus Notice)|
|   Isolates substantive legal query    |             +-----------------------+
+---------------------------------------+
                    |
                    v
+---------------------------------------------------------------------------------------+
|                                Dense Vector Retrieval                                 |
|      all-MiniLM-L6-v2 Embeddings | Scoped FAISS Index | Heading-Aware Chunks          |
+---------------------------------------------------------------------------------------+
                    |
                    v
+---------------------------------------------------------------------------------------+
|                             Calibrated Score Abstention                               |
|       Checks top similarity >= 0.30 and margin criteria against borderline noise       |
+---------------------------------------------------------------------------------------+
                    |
                    +----------------------+----------------------+
                    |                                             |
           [Score >= Threshold]                           [Score < Threshold]
                    |                                             |
                    v                                             v
+---------------------------------------+             +-----------------------+
|            Ranked Results             |             |       Abstention      |
| Source document, offsets [start:end], |             | (Absent Clause Notice)|
|       scores, and clause text         |             +-----------------------+
+---------------------------------------+
```

### Key Architectural Decisions

1. Entity-Aware Document Routing: In commercial contract analysis and in LegalBench RAG, natural language queries identify the agreement or contracting parties. The Document Router extracts entity tokens and restricts search to the target document. This eliminates cross contract interference across the 29 agreements, shrinks the search space by 96 percent, and provides instantaneous rejection for agreements absent from the corpus.
2. Query Prefix Stripping (`strip_contract_prefix`): LegalBench RAG prompts prepend contract titles before a semicolon (for example: "Consider the Strategic Alliance Agreement between X and Y; What is the governing law?"). While this prefix assists the router, passing it to the dense embedder contaminates the query representation with contract title tokens. Stripping this prefix after routing allows the embedding model to represent strictly the substantive legal question, removing preamble matching and unlocking Rank 1 accuracy.
3. Heading-Aware Chunking: Contracts are hierarchical documents divided into Articles, Sections, and Clauses. Our regex heading extractor identifies active structural headings (such as `ARTICLE 14`, `SECTION 2.1`, or `14. Governing Law`). The chunker attaches the active heading to `embedding_text` while strictly preserving verbatim raw character coordinates in `start` and `end`. The embedding model receives structural context, yet returned citations map character for character to the raw document text.
4. Word-Boundary Aligned Fixed Chunking: Sliding window chunkers often cut words arbitrarily at chunk boundaries. Our fixed segmenter word aligns chunk starts and ends using regex whitespace boundaries, ensuring chunks begin and terminate on full words.
5. Calibrated Threshold Abstention: Top candidate similarity scores are checked against a calibrated threshold (0.30 for cosine similarity). If the maximum score fails to meet the threshold, or if the candidate agreement is unindexed, the pipeline returns an empty result set and issues an explicit abstention message rather than returning irrelevant text.

### Alternatives Considered and Evaluated

1. Global Unrouted Retrieval: Searching across all 29 contracts simultaneously causes boilerplate clauses (such as standard definitions, notices, and severability) from competing agreements to displace the true answering clause. As evaluated in Section 5, disabling document routing dropped F1 at 5 from 0.1383 to 0.0261.
2. Clause-Boundary Chunking (`strategy="clause"`): Boundary-snapped clause chunking targeted 400 characters at punctuation and paragraph delimiters. In testing, this reduced recall because substantive legal covenants frequently span multi-sentence structures that smaller fragments truncated.
3. Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`): Re-ranking candidates with cross-encoders improved precision on certain clauses, but raw logit scale discrepancies and prompt title matching caused false rejections when evaluated against standard cosine thresholds.
4. Lexical BM25 Hybrid Fusion: Combining BM25 with dense retrieval via linear or reciprocal rank fusion was explored. Linear fusion suffered from score scale distortion, while reciprocal rank fusion generated fractional scores that tripped absolute score thresholds. Pure dense retrieval with query prefix stripping and heading-aware chunking outperformed hybrid variants on this benchmark.

## 3. LLM Use

No Large Language Model is utilized within the runtime inference pipeline.

The retrieval pipeline relies entirely on dense sentence transformers, FAISS vector indexing, regex entity routing, structural heading extraction, and calibrated score thresholding. This explicit architectural choice provides three benefits:
1. Complete Data Privacy: Commercial contracts containing confidential terms are processed entirely on local CPU memory without external API transmissions.
2. Zero Hallucination: The pipeline returns verbatim document spans with verified character offsets rather than generative summaries.
3. Deterministic Low Latency: Sub 30 millisecond query execution allows high throughput without token fees or rate limits.

An AI coding assistant was utilized strictly as an offline software engineering tool during development for drafting boilerplate configuration schemas, formatting Markdown documentation, and writing unit test fixtures.

## 4. Evaluation Setup

### Benchmark and Metrics

Evaluation is performed on LegalBench RAG CUAD, an objective benchmark created by legal domain experts. We adhere strictly to character-level metric definitions:

$$\text{Precision} = \frac{|\text{Retrieved Characters} \cap \text{Gold Characters}|}{|\text{Retrieved Characters}|}$$

$$\text{Recall} = \frac{|\text{Retrieved Characters} \cap \text{Gold Characters}|}{|\text{Gold Characters}|}$$

$$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Metrics are computed at cutoffs $k \in \{1, 2, 3, 4, 5, 8, 10, 16, 32, 64\}$. Overlap is calculated at the character level across matching document paths; retrieved characters from non-matching documents contribute zero overlap.

### Partitions and Query Sets

The evaluation spans 194 benchmark queries across 29 commercial contracts:
- Train Split: 140 answerable queries (72.2 percent)
- Test Split: 54 answerable queries (27.8 percent)
- Corpus Scope: 29 unique agreements totaling 2,686,303 characters

Deterministic partitioning assigns queries via SHA256 hashing:

$$h = \text{int}(\text{SHA256}(\text{query}.\text{strip}()).\text{hexdigest}(), 16)$$

$$\text{split} = \text{test if } (h \bmod 10 < 3) \text{ else train}$$

### Unanswerable Query Methodology and Full History

To evaluate abstention, two unanswerable query sets are integrated:
1. Original Unanswerable Set (`data/unanswerable_queries.json`): 35 queries authored during initial project setup. Under SHA256 partitioning, this yields 24 train and 11 test queries. This file was preserved completely intact without modifying existing lines or rewriting git history.
2. Extra Unanswerable Set (`data/unanswerable_extra.json`): 35 verified in-corpus near-miss queries authored in `commit d86951d`. These queries reference actual indexed agreements (such as BicycleTX, Papa John's, HPIL, King Pharma, Flotek, Dynamex, Duos Tech, and Vitamin Shoppe), inquiring about covenants confirmed 100 percent absent via full-text keyword verification (such as cryptocurrency executive bonuses, vegan cheese product formulas, electric vehicle charging subsidies, NFT ticketing royalties, and carbon offset covenants). Under SHA256 partitioning, this yields 23 train and 12 test queries.

Combined query counts:
- Train Split: 187 queries (140 answerable, 47 unanswerable: 24 original + 23 extra)
- Test Split: 77 queries (54 answerable, 23 unanswerable: 11 original + 12 extra)
- Total Benchmark Queries: 264 queries

Abstention is tracked across three distinct dimensions:
- Out-of-corpus Abstention: Proportion of queries referencing unindexed agreements correctly rejected
- In-corpus Near-miss Abstention: Proportion of absent clause queries referencing indexed agreements correctly rejected
- False Negative Abstention: Proportion of answerable benchmark queries incorrectly rejected

### Protocol Integrity and Test Run Disclosures

All test evaluations across repository history are logged in `logs/test_runs.jsonl`:
- Test was run 4 times in total. All 4 of those evaluations were before the v2 improvement loop, and none were used to select configurations in v2.
  1. Run 1 (2026-11-01 14:25): Baseline test evaluation on original set (65 queries, Recall@5 = 0.0780, Abstention = 0.0)
  2. Run 2 (2026-11-01 16:33): Exploratory test evaluation on larger 1024 chunks (66 queries, Recall@5 = 0.2195, Abstention = 0.1667), disclosed as exploratory peeking
  3. Run 3 (2026-11-01 18:53): Baseline re-evaluation on original set (65 queries, Recall@5 = 0.0780, Abstention = 0.0)
  4. Run 4 (2026-11-01 18:55): Evaluation of v1 final configuration on original set (65 queries, Recall@5 = 0.1036, Abstention = 0.3636)
- During the entire v2 redesign and 6 iteration improvement loop, the test split was evaluated zero times (0 test runs during tuning).

## 5. Evaluation Results

The table below contrasts all configurations evaluated on the training split (187 queries: 140 answerable, 47 unanswerable), including the realistic unrouted condition.

### Training Split Comprehensive Comparison (187 Queries)

| Configuration | P@1 | R@1 | F1@1 | P@5 | R@5 | F1@5 | R@10 | Out-of-Corpus Abstention | In-Corpus Abstention | Total Abstention | False Abstention | Rank 1 Preamble Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (v1)** | 0.0306 | 0.0262 | 0.0254 | 0.0408 | 0.1888 | 0.0617 | 0.2998 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5500 |
| **Final (v1)** | 0.0658 | 0.1327 | 0.0817 | 0.0476 | 0.4210 | 0.0816 | 0.5909 | 0.7917 | 0.2609 | 0.5319 | 0.0000 | 0.4071 |
| **Final (v2)** | **0.1571** | **0.1812** | **0.1527** | **0.0881** | **0.4504** | **0.1383** | **0.5929** | **0.7917** | **0.4783** | **0.6383** | **0.0000** | **0.0071** |
| *Realistic Unrouted* | 0.0247 | 0.0242 | 0.0215 | 0.0182 | 0.0647 | 0.0261 | 0.0998 | 0.0417 | 0.0000 | 0.0213 | 0.0000 | 0.0000 |

### Detailed Performance Across Rank Cutoffs (v2 Final on Train)

| k | Precision@k | Recall@k | F1@k |
| :--- | :--- | :--- | :--- |
| **1** | 0.1571 | 0.1812 | 0.1527 |
| **2** | 0.1457 | 0.2827 | 0.1756 |
| **3** | 0.1119 | 0.3383 | 0.1540 |
| **4** | 0.0977 | 0.3917 | 0.1446 |
| **5** | 0.0881 | 0.4504 | 0.1383 |
| **8** | 0.0687 | 0.5372 | 0.1157 |
| **10** | 0.0630 | 0.5929 | 0.1088 |
| **16** | 0.0473 | 0.6654 | 0.0852 |
| **32** | 0.0318 | 0.7945 | 0.0595 |
| **64** | 0.0220 | 0.9058 | 0.0420 |

### Analysis of Primary Findings

1. Preamble Bias Elimination: In v1, 40.7 percent of training queries retrieved Chunk 0 (the contract title, date, and recital preamble) at Rank 1. This occurred because queries include the contract title in their prompt. In v2, stripping the contract prefix after document routing reduced the Rank 1 preamble rate to 0.71 percent (1 of 140 queries). As a direct result:
   - Precision@1 increased from 0.0658 to 0.1571 (+138.8 percent relative gain)
   - Recall@1 increased from 0.1327 to 0.1812 (+36.5 percent relative gain)
   - F1@1 surged from 0.0817 to 0.1527 (+86.9 percent relative gain)
2. Heading-Aware Structural Context: Adding contractual headings (e.g. `ARTICLE`, `SECTION`) to chunk embedding representations while preserving raw offsets improved Mean F1@5 from 0.1162 to 0.1383 (+19.0 percent relative gain) and Precision@5 from 0.0681 to 0.0881 (+29.4 percent relative gain). Chunks that previously lacked semantic context now match queries targeting specific contractual topics (such as governing law or non-competes).
3. The Realistic Unrouted Condition: When document routing is disabled (simulating a search across all 29 contracts simultaneously without knowing which agreement applies), F1@5 collapses from 0.1383 down to 0.0261, and Recall@10 falls from 0.5929 to 0.0998. This occurs because boilerplate language (such as indemnities, confidentiality clauses, and definitions) from 28 competing contracts outscores the substantive clause in the target contract. Entity-aware document routing is therefore the most vital architectural component for legal RAG benchmarks.

## 6. Train-to-Test Abstention Gap Analysis

In the v1 evaluation report, unanswerable abstention was reported as 79.2 percent on train (19 of 24) versus 36.4 percent on test (4 of 11). This large discrepancy was not caused by model variance, but by an imbalance in the composition of the original unanswerable query set:

- Train Set Composition (v1): 19 of 24 queries (79.2 percent) referenced agreements outside the corpus (e.g. Netflix, Sony Pictures, Marvel Entertainment). The document router automatically rejected 100 percent of these queries because the agreement did not exist in the corpus index. Only 5 queries referenced in-corpus agreements, and the system abstained on 0 of them.
- Test Set Composition (v1): Only 4 of 11 queries (36.4 percent) referenced agreements outside the corpus. The remaining 7 queries referenced in-corpus agreements, on which the raw dense threshold of 0.30 never fired.

The apparent 79.2 percent versus 36.4 percent abstention rate was an artifact of the router's out-of-corpus matching rule. The dense similarity threshold was never rejecting absent clauses within valid contracts.

To resolve this distortion, we created `data/unanswerable_extra.json` containing 35 verified in-corpus near-misses (23 train, 12 test). In v2, in-corpus near-miss abstention is measured directly:
- v1 In-Corpus Abstention: 26.09 percent (6 of 23 queries rejected)
- v2 In-Corpus Abstention: 47.83 percent (11 of 23 queries rejected)
- v2 Out-of-Corpus Abstention: 79.17 percent (19 of 24 queries rejected)
- v2 Total Abstention: 63.83 percent (30 of 47 queries rejected)

By decoupling in-corpus near-miss abstention from out-of-corpus router rejection, the evaluation pipeline provides an honest measurement of absent clause detection.

## 7. Self-Improvement Loop Iteration History

The improvement engine executed 13 systematic iterations across repository history (Iterations 0 to 6 in v1, and Iterations 7 to 12 in v2). All iterations were evaluated exclusively on the train split against predefined guardrails:
- Primary Optimization Metric: Mean F1 at 5 on train
- Guardrail 1: Relative drop in Precision at 5 must not exceed 10 percent
- Guardrail 2: False negative abstention on answerable queries must not exceed 0.15
- Guardrail 3: In-corpus unanswerable abstention must not decrease
- Minimum Primary Gain: F1 at 5 gain must be at least 0.005 to accept

### Complete Log of All Iterations (0 through 12)

| Iteration | Run Label | Proposal Name | Proposed By | R@5 | P@5 | F1@5 | In-Corpus Abstention | Decision | Core Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | v1 | `baseline` | rule | 0.1888 | 0.0408 | 0.0617 | 0.0000 | Baseline | Initial unrouted dense baseline with fixed 512 chunks and raw 0.30 threshold |
| **1** | v1 | `enable_document_routing` | rule | 0.2393 | 0.0488 | 0.0750 | 0.0000 | ACCEPTED | Scoped search to referenced agreement, eliminating cross-contract false positive noise |
| **2** | v1 | `enable_clause_chunking` | rule | 0.1815 | 0.0555 | 0.0722 | 0.0000 | REJECTED | Target 400 char clause boundaries fractured multi-sentence covenants, reducing recall |
| **3** | v1 | `enable_cross_encoder_reranker` | rule | 0.1562 | 0.0287 | 0.0435 | 0.0000 | REJECTED | Prompt title matching and uncalibrated cross-encoder logits distorted ranking |
| **4** | v1 | `enable_bm25_hybrid` | rule | 0.1744 | 0.0319 | 0.0484 | 0.0000 | REJECTED | Whitespace split tokenization and prompt title tokens dominated BM25 scoring |
| **5** | v1 | `increase_chunk_overlap` | rule | 0.2355 | 0.0516 | 0.0768 | 0.0000 | REJECTED | Recall at 5 dropped slightly from 0.2393 to 0.2355; gain below threshold |
| **6** | v1 | `tune_dense_threshold_0.35` | rule | 0.2393 | 0.0488 | 0.0750 | 0.0000 | REJECTED | Raising threshold to 0.35 produced zero change in recall or abstention |
| **7** | v2 | `strip_query_prefix` | diagnosis | 0.4357 | 0.0681 | 0.1162 | 0.4783 | ACCEPTED | Stripped agreement prefix from query, reducing rank 1 preamble from 40.7% to 2.1% |
| **8** | v2 | `heading_aware_chunking` | diagnosis | 0.4504 | 0.0881 | 0.1383 | 0.4783 | ACCEPTED | Prepended structural article and section headers, boosting F1@5 by +0.0221 |
| **9** | v2 | `bm25_rrf_hybrid` | parameter_search | 0.0000 | 0.0000 | 0.0000 | 1.0000 | REJECTED | RRF scores (~0.016) fell below 0.30 threshold, tripping 15% false abstention guardrail |
| **10** | v2 | `tune_chunk_size_384` | parameter_search | 0.4412 | 0.0863 | 0.1345 | 0.4783 | REJECTED | F1@5 dropped from 0.1383 to 0.1345; failed minimum gain criterion |
| **11** | v2 | `tune_score_threshold_0.42` | diagnosis | 0.4017 | 0.0807 | 0.1257 | 0.7391 | REJECTED | In-corpus abstention reached 73.9%, but F1@5 dropped by 0.0125 and false abstention was 11.4% |
| **12** | v2 | `tune_margin_threshold_0.02` | parameter_search | 0.2575 | 0.0553 | 0.0850 | 0.8696 | REJECTED | Tripped false abstention guardrail with 50.0% false abstention on answerable queries |

The self-improvement loop converged on `iter_8_heading_aware_chunking` as the definitive v2 configuration, preserved in `configs/final_v2.yaml`.

## 8. Failure Diagnosis

Systematic diagnosis of the v2 final configuration across the 187 training queries categorized all queries into six outcomes:
- Perfect Match: 51 queries (27.3 percent)
- Wrong Section: 61 queries (32.6 percent)
- Partial Overlap: 28 queries (15.0 percent)
- True Abstention (Out-of-corpus): 19 queries (10.2 percent)
- True Abstention (In-corpus Near-miss): 11 queries (5.9 percent)
- False Retrieval (Out-of-corpus): 5 queries (2.7 percent)
- False Retrieval (In-corpus Near-miss): 12 queries (6.4 percent)
- False Abstention on Answerable: 0 queries (0.0 percent)

Below are 16 concrete failure cases analyzed with queries, ground truth spans, retrieved spans, and root cause diagnosis.

### Category 1: Section Misalignment Within Correct Document (Wrong Section)

1. Query: "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; Are there any exceptions to competitive restrictions in this contract?"
   - Gold Span: `[33572:34034]` in `BICYCLETHERAPEUTICSPLC...txt` (Section 15.3: Permitted shareholding exceptions)
   - Retrieved Top 1: `[38350:38858]` (Section 17: Post-termination obligations)
   - Retrieved Score: 0.5144
   - Root Cause: Section 17 discusses restrictive covenant duration and termination remedies, exhibiting strong semantic similarity with "competitive restrictions", while Section 15.3 is a narrow financial exception.
2. Query: "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; Does this contract include any right of first refusal, right of first offer, or right of first negotiation?"
   - Gold Span: `[6235:6531]` and `[6533:7937]` in `BICYCLETHERAPEUTICSPLC...txt` (Section 4: Option grants and vesting)
   - Retrieved Top 1: `[55758:56267]` (Schedule 2: Statutory Employment Terms)
   - Retrieved Score: 0.4415
   - Root Cause: Schedule 2 details employee rights upon transfer of undertakings, drawing vector similarity against generic contractual "rights" terminology.
3. Query: "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; How is intellectual property ownership assigned in this contract?"
   - Gold Span: `[44840:45359]` in `BICYCLETHERAPEUTICSPLC...txt` (Section 19: Intellectual Property Inventions Assignment)
   - Retrieved Top 1: `[48408:48917]` (Section 21: Company property return covenants)
   - Retrieved Score: 0.4972
   - Root Cause: Section 21 contains references to return of confidential documents and company property, competing with the IP assignment clause.
4. Query: "Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; Is there an anti-assignment clause in this contract?"
   - Gold Span: `[17341:17432]` in `HPILHOLDING...txt` (Section 17: Assignment prohibition)
   - Retrieved Top 1: `[19159:19669]` (Section 22: Execution and delivery)
   - Retrieved Score: 0.4142
   - Root Cause: The gold anti-assignment clause is short (91 characters), while Section 22 contains general corporate authority language that scores higher in raw cosine similarity.
5. Query: "Consider the Service Agreement between Theravance Biopharma UK Limited and Brett Haumann; Is there a clause preventing the solicitation of employees in this contract?"
   - Gold Span: `[25311:25713]` in `THERAVANCEBIOPHARMA...txt` (Section 12: Non-solicitation of personnel)
   - Retrieved Top 1: `[32256:32768]` (Section 18: Severability and survival)
   - Retrieved Score: 0.4352
   - Root Cause: Survival clauses enumerate surviving sections by number without mentioning "solicitation" explicitly, causing dense embeddings to misalign.

### Category 2: Partial Overlap Across Extended Clauses

6. Query: "Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; What is the expiration date of this contract?"
   - Gold Span: `[5604:5739]` in `HPILHOLDING...txt` (135 characters)
   - Retrieved Top 1: `[5639:6151]` (512 characters)
   - Overlap: 100 characters captured out of 135 gold characters (Recall = 0.7407)
   - Root Cause: The sliding window step landed at character 5639, capturing the termination year and notice provisions while cutting the first 35 characters of the commencement sentence.
7. Query: "Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; Is there a cap on liability under this contract?"
   - Gold Spans: `[11380:11583]` and `[12564:12768]` in `HPILHOLDING...txt`
   - Retrieved Top 1: `[18004:18509]`
   - Retrieved Score: 0.4731
   - Root Cause: The agreement distributes liability caps across multiple distinct subclauses. The chunker retrieved the broad indemnification limit rather than the direct damage exclusion.
8. Query: "Consider the Sponsorship Agreement between Jacksonville Jaguars, LLC and The ARC Group, Inc.; What licenses are granted under this contract?"
   - Gold Span: `[1697:2488]` in `ArcGroupInc...txt` (791 characters)
   - Retrieved Top 1: `[2332:2844]` (512 characters)
   - Overlap: 156 characters captured (Recall = 0.1972)
   - Root Cause: The license grant is a multi-paragraph provision of 791 characters. A single 512 character chunk covers only the tail end of the trademark grant.
9. Query: "Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; How is intellectual property ownership assigned in this contract?"
   - Gold Spans: `[14401:14603]`, `[14605:15442]`, `[15667:16433]` in `DUOSTECHNOLOGIESGROUP...txt`
   - Retrieved Top 2: `[15901:16411]`
   - Overlap: Partial recall on the third span, missing earlier definition spans.
   - Root Cause: The gold annotation encompasses three distinct subsections spanning 2,032 characters, exceeding the window of a single chunk.
10. Query: "Consider the Service Agreement between Theravance Biopharma UK Limited and Brett Haumann; Is there a covenant not to sue included in this contract?"
    - Gold Span: `[28110:28650]` in `THERAVANCEBIOPHARMA...txt` (540 characters)
    - Retrieved Top 1: `[28160:28672]`
    - Overlap: 490 of 540 characters captured (Recall = 0.9074)
    - Root Cause: Word alignment snapped chunk start to the second sentence of the mutual release, missing the introductory preamble of the clause.

### Category 3: False Retrieval on In-Corpus Near-Misses (Absent Clauses)

11. Query: "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; Does this contract specify requirements for cryptocurrency payment of executive bonuses?"
    - Gold Span: None (Unanswerable near-miss)
    - Retrieved Top 1: `[3870:4382]` (Section 3: Executive Compensation and Discretionary Bonus)
    - Retrieved Score: 0.5059 (surpassed 0.30 threshold)
    - Root Cause: The contract contains extensive bonus and salary provisions. Dense embeddings match "bonus" and "executive payment" strongly, scoring 0.5059 despite the complete absence of "cryptocurrency".
12. Query: "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; What are the requirements for company car electric vehicle charging station subsidies in this contract?"
    - Gold Span: None (Unanswerable near-miss)
    - Retrieved Top 1: `[7757:8262]` (Section 5: Travel and Business Expense Reimbursement)
    - Retrieved Score: 0.3694 (surpassed 0.30 threshold)
    - Root Cause: General business expense reimbursement provisions match vehicle and travel queries above the baseline threshold.
13. Query: "Consider the Sponsorship Agreement between Jacksonville Jaguars, LLC and The ARC Group, Inc.; What is the revenue share percentage for commemorative NFT ticketing sales?"
    - Gold Span: None (Unanswerable near-miss)
    - Retrieved Top 1: `[5432:5942]` (Section 4: Ticket Allocation and Hospitality Benefits)
    - Retrieved Score: 0.3889 (surpassed 0.30 threshold)
    - Root Cause: Ticket distribution clauses share semantic overlap with ticketing sales inquiries, scoring above 0.30 in the absence of an NFT specific filter.
14. Query: "Consider the Endorsement Agreement between Papa John's International, Inc. and Shaquille O'Neal; What are the vegan cheese product formulation approval procedures in this agreement?"
    - Gold Span: None (Unanswerable near-miss)
    - Retrieved Top 1: `[39488:39994]` (Section 8: Product Quality and Ingredient Approvals)
    - Retrieved Score: 0.4832 (surpassed 0.30 threshold)
    - Root Cause: The endorsement agreement contains authentic quality control and ingredient approval covenants, creating high semantic similarity with specialized formulation queries.

### Category 4: False Retrieval on Out-of-Corpus Agreements

15. Query: "Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; Is there a carbon credit offset obligation or environmental sustainability covenant in this contract?"
    - Gold Span: None (Unanswerable)
    - Retrieved Top 1: `[3317:3822]` (Section 2: Renewable Energy Project Development Recitals)
    - Retrieved Score: 0.4084 (surpassed 0.30 threshold)
    - Root Cause: The agreement repeatedly references solar and wind project development, producing contextual overlap with environmental covenants.
16. Query: "Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; What is the maximum cap on consequential damages for gross negligence?"
    - Gold Span: None (Unanswerable)
    - Retrieved Top 1: `[19363:19874]` (Section 8: General Limitation of Liability)
    - Retrieved Score: 0.3456 (surpassed 0.30 threshold)
    - Root Cause: General liability exclusion provisions score 0.3456 because they discuss indirect damages and negligence, even though gross negligence carve-outs are absent.

## 9. Production Deployment Considerations

### Latency Profiles and Throughput

- In warm execution, single-query inference averages 27.9 milliseconds for routed retrieval on CPU, compared to 31.0 milliseconds for unrouted search.
- FAISS inner product search over the routed contract partition executes in under 2 milliseconds; query embedding generation via `all-MiniLM-L6-v2` accounts for the remaining 26 milliseconds.
- High throughput concurrent production workloads can batch queries behind an ASGI framework (such as FastAPI) on GPU or multi-core CPU instances, achieving > 300 queries per second.

### Memory Footprint and Index Caching

- Across 29 contracts, 3,139 heading-aware chunks are generated. At 384 dimensions in float32, the FAISS vector index requires 4.8 megabytes of RAM.
- Chunk text metadata and routing tables require 7.2 megabytes.
- Total resident memory per worker process is under 150 megabytes (including PyTorch and sentence transformer model weights).
- For enterprise repositories of 100,000 contracts (approximately 20 million chunks), the FAISS index occupies 31 gigabytes, which can be sharded across nodes or compressed via FAISS IVFPQ (inverted file product quantization) to under 8 gigabytes.

### Partition-Level Updates Without Full Re-Indexing

In commercial legal environments, contracts are amended, added, or terminated continuously:
- The modular architecture supports partition-level updates: each agreement is indexed as an independent chunk partition keyed by document path.
- When an agreement is modified, only its specific chunks are re-embedded and updated in the FAISS index using `faiss.IndexIDMap2`, without perturbing other indexed contracts.
- Document routing dictionaries are updated in $O(1)$ time by registering new contract title aliases and party entities.

### Multi-Tenant Isolation and Authorization

- In corporate media organizations (such as JioStar content rights management), contract access must enforce strict authorization boundaries (such as sports broadcast licenses versus international studio distribution agreements).
- Entity-aware document routing naturally enforces tenant isolation by restricting candidate retrieval to authorized contract partitions matching user credentials.

## 10. Limitations and Strategic Recommendations

### Current Limitations

1. Vocabulary Granularity of General Embeddings: `all-MiniLM-L6-v2` is pretrained on general web text. While it handles semantic similarity effectively, it does not distinguish subtle legal nuances between closely related covenants (for example, general non-competes versus carve-outs for passive minority investments).
2. Extended Clause Fragmentation: Fixed 512 character sliding windows inevitably truncate long contractual provisions that span multiple subsections.
3. In-Corpus Near-Miss Discrimination: When an agreement contains extensive provisions regarding a topic (such as executive bonuses or licensing), questions asking about absent subclauses (such as cryptocurrency bonuses) receive similarity scores above 0.30 due to shared contextual vocabulary.

### Strategic Recommendations

1. Domain-Adapted Legal Sentence Embedding Fine-Tuning: Fine-tune bi-encoders using contrastive loss on legal clause pairs (such as CUAD and SEC EDGAR filings) with hard negative mining to distinguish operational clauses from boilerplate recitals.
2. Hierarchical Clause Expansion: Store contract structural hierarchy (Article $\rightarrow$ Section $\rightarrow$ Subsection). When a chunk is retrieved, dynamically expand the returned character span to encompass the complete parent subsection.
3. Two-Stage Calibrated Cross-Encoder with Margin Filtering: Deploy a specialized legal cross-encoder on the top 15 candidates with sigmoid probability calibration to enforce sharp rejection of in-corpus near-misses.

## 11. Verification and User Execution Command

To verify the test split strictly once on the final v2 configuration, execute the following command:

```bash
python run.py evaluate --config configs/final_v2.yaml --split test
```

This will run evaluation across all 77 test queries (54 answerable, 23 unanswerable), compute character-level precision, recall, and F1 across all cutoffs $k \in \{1, 2, 3, 4, 5, 8, 10, 16, 32, 64\}$, and append an auditable entry to `logs/test_runs.jsonl`.
