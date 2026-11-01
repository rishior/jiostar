# Evaluation Report: Contract Intelligence Assistant

## 1. Executive Summary

We developed an automated legal clause retrieval and verification system evaluated on the LegalBench RAG CUAD benchmark. The assistant accepts natural language inquiries regarding commercial contract covenants, rights, and restrictions, returning ranked passages with exact document paths and character offsets, or abstaining when an inquiry cannot be answered from the contract corpus.

Our final architecture implements entity aware document routing combined with 512 character sliding window chunking, dense vector retrieval using normalized sentence embeddings (`all-MiniLM-L6-v2`), and calibrated cosine thresholding. 

On the held out test partition (65 total queries comprising 54 answerable contract questions and 11 unanswerable queries), our final system achieved:
- Headline Recall at 5 of 0.1036 (up from 0.0780 in the baseline, a 32.8 percent relative gain)
- Recall at 10 of 0.3347 (up from 0.2028, a 65.0 percent relative gain)
- Precision at 5 of 0.0190 (up from 0.0167, a 13.8 percent relative gain)
- F1 at 5 of 0.0292 (up from 0.0250, a 16.8 percent relative gain)
- True Negative Abstention rate on unanswerable queries of 36.4 percent (4 out of 11 rejected, up from 0.0 percent in the baseline)
- Zero false abstentions on answerable queries across both train and test sets
- Query latency of 58 milliseconds per query (down from 280 milliseconds, a 4.8x acceleration)

All parameter tuning, failure analysis, and iterative proposals were developed strictly on the train partition. The test partition was evaluated strictly once for final reporting.

## 2. System Architecture

The architecture prioritizes character level fidelity, modular component isolation, and deterministic execution.

### Key Architectural Decisions

1. Entity Aware Document Routing: Legal contract benchmarks and commercial workflows typically specify the target agreement or party identities within the query prompt. By matching query tokens against canonical contract titles, filenames, and party profiles, the retriever restricts dense search to the relevant agreement. This eliminates cross contract false positive noise, shrinks candidate vector space by 96 percent, and allows immediate abstention for agreements outside the corpus.
2. Character Offset Preserving Segmenter: Commercial legal analysis requires exact pinpoint citations. Standard text segmenters compute offsets heuristically or lose them during tokenization. Our chunkers preserve exact bidirectional character coordinates in UTF8 text mode without newline normalization, guaranteeing that slicing `document[start:end]` returns the exact passage.
3. Dense Vector Indexing via Inner Product Search: Text chunks are embedded into 384 dimensional normalized vectors using `all-MiniLM-L6-v2`. Cosine similarity is computed via FAISS inner product search, supporting instantaneous scoped filtering per document.
4. Calibrated Threshold Abstention: When the top candidate similarity score falls below 0.30, or when the query references an unindexed agreement, the system returns an empty result set accompanied by an explicit abstention notice rather than returning irrelevant text.

### Alternatives Considered and Evaluated

1. Unrouted Global Search: The baseline searched across all corpus contracts simultaneously. While functional, top candidates from unrelated contracts frequently outscored the true clause due to generic boilerplate vocabulary (for example, definitions of "affiliates" or "indemnification").
2. Clause Aware Semantic Segmentation: We implemented structural chunking snapping to numbered clause headers and paragraph breaks (target 400 characters). On the train split, Recall at 5 dropped from 0.2393 to 0.1815 because legal questions match broader contextual provisions that smaller fragments truncated.
3. Cross Encoder Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`): We evaluated a two stage pipeline reranking the top 30 dense candidates with a cross encoder. Although it achieved 83.3 percent true negative abstention on unanswerable queries, Recall at 5 dropped to 0.1562. The MS MARCO weights, trained on general web search, penalized legal boilerplate phrases containing valid legal answers.
4. Sparse Lexical BM25 Hybrid Fusion: Combining BM25 with dense retrieval via linear score fusion reduced Recall at 5 from 0.2393 to 0.1744. Legal contracts share repetitive formal terminology across clauses, causing lexical search to rank unrelated boilerplate containing exact keyword repetitions above substantive answers.

## 3. LLM Use

We did not use a Large Language Model within the runtime inference pipeline.

The retrieval pipeline relies on dense sentence transformers, FAISS indexing, document routing, and calibrated score thresholding. This explicit design decision was guided by three considerations:

1. Data Confidentiality and Enterprise Privacy: Commercial media agreements contain confidential licensing terms, distribution guarantees, minimum revenue thresholds, and exclusivity covenants. Transmitting contract text to third party LLM APIs introduces data leakage risks and regulatory liabilities for an enterprise media organization.
2. Inference Latency and Operational Footprint: The dense retrieval system operates in under 60 milliseconds per query on local CPU hardware without GPU acceleration or cloud API dependencies.
3. Hallucination Elimination and Legal Auditability: Legal rights administrators require verbatim contractual clauses and source coordinates. Generative language models risk hallucinating terms or altering legal obligations. Direct passage retrieval preserves evidentiary auditability.

An AI coding assistant was utilized strictly as an offline software engineering aid for generating boilerplate test fixtures, formatting Markdown tables, and drafting synthetic out of domain queries.

## 4. Evaluation Setup

### Benchmark and Metrics

Evaluation is performed on LegalBench RAG CUAD, an objective benchmark annotated by legal professionals. We adhere strictly to the character level metric definitions established in the LegalBench RAG specification:

$$\text{Precision} = \frac{|\text{Retrieved Characters} \cap \text{Gold Characters}|}{|\text{Retrieved Characters}|}$$

$$\text{Recall} = \frac{|\text{Retrieved Characters} \cap \text{Gold Characters}|}{|\text{Gold Characters}|}$$

$$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Metrics are evaluated at rank cutoffs $k \in \{1, 3, 5, 10\}$. Overlap is calculated at the character level across matching document paths. Characters from non matching documents contribute zero overlap.

### Sample Sizes and Partition Methodology

The dataset comprises 194 CUAD benchmark queries selected via the canonical mini function. Using the deterministic SHA256 query hash partition:

$$h = \text{int}(\text{SHA256}(\text{query}.\text{strip}()).\text{hexdigest}(), 16)$$

$$\text{split} = \text{test if } (h \bmod 10 < 3) \text{ else train}$$

The partition yields:
- Train Split: 140 answerable queries (72.2 percent)
- Test Split: 54 answerable queries (27.8 percent)
- Corpus Scope: 29 unique commercial agreements totaling 2,686,303 characters

### Unanswerable Query Methodology

To rigorously evaluate abstention, we authored 35 out of domain and adversarial unanswerable queries. These include realistic near miss queries regarding absent covenants (such as cryptocurrency settlements, GDPR data sovereignty, carbon offset mandates, and drone transit) as well as questions naming entities outside the contract corpus.

Applying the identical SHA256 partition yields:
- Train Unanswerable: 24 queries (68.6 percent)
- Test Unanswerable: 11 queries (31.4 percent)

Abstention is measured along two complementary axes:
- True Negative Abstention Rate: Proportion of unanswerable queries where the system correctly abstains.
- False Negative Abstention Rate: Proportion of answerable benchmark queries where the system incorrectly abstains.

### Protocol Integrity and Test Run Count

To prevent data snooping and overfitting:
- All exploratory analysis, failure diagnosis, and configuration proposals were executed exclusively on the train split.
- The test split was evaluated strictly once on the baseline configuration and strictly once on the final configuration, for a total test run count of 1 for final verification.
- Zero test split runs were used to select or tune configurations.

## 5. Evaluation Results

The tables below contrast the Baseline configuration against the Final configuration across both splits.

### Train Split Evaluation (140 Answerable, 24 Unanswerable)

| Metric | Baseline (512 / 128, Unrouted) | Final (512 / 128, Routed) | Absolute Change | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| Precision at 1 | 0.0306 | 0.0306 | 0.0000 | 0.0% |
| Recall at 1 | 0.0262 | 0.0262 | 0.0000 | 0.0% |
| F1 at 1 | 0.0254 | 0.0254 | 0.0000 | 0.0% |
| Precision at 3 | 0.0435 | 0.0465 | +0.0030 | +6.9% |
| Recall at 3 | 0.1257 | 0.1386 | +0.0129 | +10.3% |
| F1 at 3 | 0.0582 | 0.0629 | +0.0047 | +8.1% |
| Precision at 5 | 0.0408 | 0.0488 | +0.0080 | +19.6% |
| Recall at 5 (Primary) | 0.1888 | 0.2393 | +0.0505 | +26.7% |
| F1 at 5 | 0.0617 | 0.0750 | +0.0133 | +21.6% |
| Precision at 10 | 0.0326 | 0.0427 | +0.0101 | +31.0% |
| Recall at 10 | 0.2998 | 0.3918 | +0.0920 | +30.7% |
| F1 at 10 | 0.0560 | 0.0734 | +0.0174 | +31.1% |
| False Negatives (Answerable Abstain) | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| True Negatives (Unanswerable Abstain) | 0.0000 | 0.7917 | +0.7917 | Infinite |
| Inference Time | 12.2s | 5.4s | -6.8s | -55.7% |

### Test Split Evaluation (54 Answerable, 11 Unanswerable)

| Metric | Baseline (512 / 128, Unrouted) | Final (512 / 128, Routed) | Absolute Change | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| Precision at 1 | 0.0018 | 0.0018 | 0.0000 | 0.0% |
| Recall at 1 | 0.0008 | 0.0008 | 0.0000 | 0.0% |
| F1 at 1 | 0.0011 | 0.0011 | 0.0000 | 0.0% |
| Precision at 3 | 0.0198 | 0.0262 | +0.0064 | +32.3% |
| Recall at 3 | 0.0465 | 0.0907 | +0.0442 | +95.1% |
| F1 at 3 | 0.0239 | 0.0351 | +0.0112 | +46.9% |
| Precision at 5 | 0.0167 | 0.0190 | +0.0023 | +13.8% |
| Recall at 5 (Primary) | 0.0780 | 0.1036 | +0.0256 | +32.8% |
| F1 at 5 | 0.0250 | 0.0292 | +0.0042 | +16.8% |
| Precision at 10 | 0.0212 | 0.0320 | +0.0108 | +50.9% |
| Recall at 10 | 0.2028 | 0.3347 | +0.1319 | +65.0% |
| F1 at 10 | 0.0352 | 0.0548 | +0.0196 | +55.7% |
| False Negatives (Answerable Abstain) | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| True Negatives (Unanswerable Abstain) | 0.0000 | 0.3636 | +0.3636 | Infinite |
| Inference Time | 18.2s | 3.8s | -14.4s | -79.1% |

## 6. Iteration History

The self improvement loop executed 6 systematic diagnostic iterations beyond the baseline. All proposals were evaluated against predefined guardrails on the train split:
- Primary Optimization Target: Maximize Mean Recall at 5
- Guardrail 1: Mean Precision at 5 drop must not exceed 0.05
- Guardrail 2: False negative abstention on answerable queries must remain below 0.15
- Guardrail 3: True negative abstention on unanswerable queries must remain above 0.50

### Summary Table of Iterations

| Iteration | Proposal Name | Primary Hypothesis | Recall at 5 | Precision at 5 | Unanswerable Abstention | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | baseline | Initial dense vector search with fixed 512 chunks and raw cosine threshold 0.30 | 0.1888 | 0.0408 | 0.0000 | Baseline |
| 1 | enable_document_routing | Constrain candidate search to referenced agreement to eliminate cross contract noise | 0.2393 | 0.0488 | 0.7917 | ACCEPTED |
| 2 | enable_clause_chunking | Target 400 character clause chunks snapping to numbered headers and paragraph breaks | 0.1815 | 0.0555 | 0.7917 | REJECTED |
| 3 | enable_cross_encoder_reranker | Rerank top 30 dense candidates with cross encoder to separate terms from boilerplate | 0.1562 | 0.0287 | 0.8333 | REJECTED |
| 4 | enable_bm25_hybrid | Combine sparse BM25 scoring with dense retrieval to capture exact section tokens | 0.1744 | 0.0319 | 0.7917 | REJECTED |
| 5 | increase_chunk_overlap | Increase overlap from 128 to 256 characters (50 percent) to resolve boundary fractures | 0.2355 | 0.0516 | 0.7917 | REJECTED |
| 6 | tune_dense_threshold_0.35 | Raise cosine abstention threshold from 0.30 to 0.35 to reject borderline negatives | 0.2393 | 0.0488 | 0.7917 | REJECTED |

### Detailed Rationale and Findings per Iteration

- Iteration 1 (enable_document_routing): Accepted. Error diagnosis on baseline revealed that 0 out of 164 queries lacked contract cues, yet unrouted search allowed unrelated contracts to contaminate top ranks. Routing improved Recall at 5 from 0.1888 to 0.2393 (+0.0505) and boosted unanswerable abstention from 0.0000 to 0.7917 by immediately abstaining on out of corpus contracts.
- Iteration 2 (enable_clause_chunking): Rejected. We hypothesized that clause boundaries would isolate concise legal definitions. However, Recall at 5 fell from 0.2393 to 0.1815 (-0.0578). Many CUAD questions match broad covenants whose meaning is fractured by smaller 400 character fragments.
- Iteration 3 (enable_cross_encoder_reranker): Rejected. Reranking top 30 candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` improved true negative abstention to 83.3 percent, but degraded Recall at 5 to 0.1562 (-0.0831). The reranker penalizes legal jargon without task specific domain adaptation.
- Iteration 4 (enable_bm25_hybrid): Rejected. We hypothesized BM25 would assist with exact section titles. Instead, Recall at 5 dropped to 0.1744 (-0.0649). Repetitive boilerplate tokens in commercial contracts distorted sparse keyword weights.
- Iteration 5 (increase_chunk_overlap): Rejected. Diagnosis identified 46 queries with partial overlap. Increasing overlap to 256 characters slightly improved Precision at 5 (from 0.0488 to 0.0516) and Recall at 10 (from 0.3918 to 0.4019), but slightly reduced Recall at 5 from 0.2393 to 0.2355 (-0.0038).
- Iteration 6 (tune_dense_threshold_0.35): Rejected. Raising the threshold to 0.35 produced no change in Recall at 5 (0.2393) and no improvement in unanswerable rejection, confirming that 0.30 is well calibrated.

The loop converged on `iter_1_document_routing` as the optimal configuration.

## 7. Failure Diagnosis

Systematic diagnosis of the final configuration across the training partition categorized 164 queries into distinct failure modes:
- Perfect Match: 26 queries (15.9 percent)
- Wrong Document: 0 queries (0.0 percent, completely resolved by router)
- Wrong Section: 74 queries (45.1 percent)
- Partial Overlap: 40 queries (24.4 percent)
- Incorrect Retrieval on Unanswerable: 5 queries (3.0 percent)
- Incorrect Abstention: 0 queries (0.0 percent)

Below are ten concrete failure examples analyzed by failure category and root cause.

### Category 1: Section Misalignment Within Correct Document (Wrong Section)

In these cases, the router identified the correct agreement, but dense similarity prioritized an unrelated contractual section containing overlapping vocabulary.

1. Query: "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; Are there any exceptions to competitive restrictions in this contract?"
   - Gold Span: `[33572:34034]` in `BICYCLETHERAPEUTICSPLC...txt` (Section 15.3: Permitted Investments Carve Out)
   - Top Retrieved Span: `[0:504]` and `[41472:41979]` (Preamble and Termination Notice Provisions)
   - Root Cause: The preamble and termination sections describe business duties and restrictions in general terms, achieving higher generic semantic overlap than the specific subclause detailing permitted investment exceptions.

2. Query: "Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; What is the governing law for this contract?"
   - Gold Span: `[16867:17117]` in `HPILHOLDING...txt` (Section 14: Applicable Law and Jurisdiction)
   - Top Retrieved Span: `[3456:3963]` and `[5760:6265]` (Section 4: Deliverables and Project Implementation)
   - Root Cause: Section 4 discusses project execution within Switzerland, confusing the semantic matcher looking for governing legal jurisdiction.

3. Query: "Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; Is there an anti-assignment clause in this contract?"
   - Gold Span: `[17341:17432]` in `HPILHOLDING...txt` (Section 17: Assignment Prohibitions)
   - Top Retrieved Span: `[3456:3963]` (Section 4: Deliverables and Technology Rights)
   - Root Cause: Section 4 repeatedly references transfer of project documentation, which exhibits dense similarity with queries concerning assignment rights.

4. Query: "Consider the Service Agreement between Theravance Biopharma UK Limited and Brett Haumann; Is there a clause preventing the solicitation of employees in this contract?"
   - Gold Span: `[25311:25713]` in `THERAVANCEBIOPHARMA...txt` (Section 12: Non Solicitation of Personnel)
   - Top Retrieved Span: `[0:512]` and `[32256:32768]` (Preamble Definitions and Severability)
   - Root Cause: Preamble definitions define "Employee" and "Executive", drawing semantic similarity while the actual non solicitation prohibition appears later under restrictive covenants.

5. Query: "Consider the Sponsorship Agreement between Excite, Inc. and Vitamin Shoppe Industries Inc. for Promotion on WebCrawler Channels; What is the expiration date of this contract?"
   - Gold Span: `[33237:33842]` in `VITAMINSHOPPECOMINC...txt` (Section 8: Term and Expiration)
   - Top Retrieved Span: `[9216:9728]` and `[2304:2814]` (Section 3: Advertising Placement Schedules)
   - Root Cause: Advertising placement schedules specify quarterly date ranges and launch milestones, which match temporal query terms more strongly than the legal contract term clause.

6. Query: "Consider the Sponsorship Agreement between Excite, Inc. and Vitamin Shoppe Industries Inc. for Promotion on WebCrawler Channels; What is the governing law for this contract?"
   - Gold Span: `[58417:58720]` in `VITAMINSHOPPECOMINC...txt` (Section 17: Governing Law)
   - Top Retrieved Span: `[9216:9728]` (Section 3: Advertising Compliance and Regulatory Obligations)
   - Root Cause: Section 3 contains references to legal compliance and governmental regulations, attracting vector similarity over the formal miscellaneous governing law provision.

### Category 2: Partial Overlap Across Extended Clauses

In these instances, the retriever returned the correct clause, but because legal rights span multiple subclauses, character recall remained incomplete.

7. Query: "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; What is the governing law for this contract?"
   - Result: Character Recall = 0.3649 (36.5 percent)
   - Gold Span: `[53956:54252]`
   - Root Cause: The governing law section couples choice of law with an exclusive dispute submission covenant. The top retrieved 512 character chunk truncated after the jurisdiction grant, omitting the arbitration venue terms.

8. Query: "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer Position; Does this contract include any right of first refusal, right of first offer, or right of first negotiation?"
   - Result: Character Recall = 0.3959 (39.6 percent)
   - Gold Span: Multi paragraph section spanning 1,450 characters
   - Root Cause: The right of first refusal spans preliminary notice, match period, and counter offer procedures. A single 512 character chunk covers only the initial grant, yielding partial recall.

9. Query: "Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; Is there a cap on liability under this contract?"
   - Result: Character Recall = 0.4988 (49.9 percent)
   - Gold Span: `[15234:16045]`
   - Root Cause: The liability limitation clause contains a monetary cap paragraph followed by an exclusion of indirect and consequential damages. The retrieved passage captured the monetary cap but missed the exclusions paragraph.

10. Query: "Consider the Service Agreement between Theravance Biopharma UK Limited and Brett Haumann; Is there a covenant not to sue included in this contract?"
    - Result: Character Recall = 0.6282 (62.8 percent)
    - Gold Span: `[28110:28650]`
    - Root Cause: The covenant not to sue is embedded inside a comprehensive mutual release of claims. The chunk captured 62.8 percent of the mutual release span before reaching the 512 character boundary.

### Category 3: False Positive Retrieval on Unanswerable Queries

11. Query: "Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG for Renewable Energy Projects; Is there a carbon credit offset obligation or environmental sustainability covenant in this contract?"
    - Top Retrieved Passage: Section 4 (`[3456:3963]`), Score = 0.6190 (surpassed threshold 0.3000)
    - Root Cause: The agreement concerns renewable energy projects (photovoltaic and wind), causing extensive contextual overlap with environmental keywords despite the absence of a carbon credit covenant.

12. Query: "Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; What is the maximum cap on consequential damages for gross negligence?"
    - Top Retrieved Passage: Section 12 (`[31872:32384]`), Score = 0.5206 (surpassed threshold 0.3000)
    - Root Cause: General limitation of liability clauses discuss consequential damages and negligence exclusions, creating strong dense similarity even when gross negligence carve outs are omitted.

## 8. Production Deployment Considerations

### Latency Profiles and Throughput

- In our production evaluation, single query latency averaged 58 milliseconds on commodity x86 CPUs without GPU acceleration.
- Document routing reduces vector comparison overhead by 96 percent by narrowing the search space from 6,000 global chunks to 200 contract specific chunks per query.
- FAISS inner product search over 200 normalized vectors executes in under 2 milliseconds; embedding generation via MiniLM accounts for the remaining 56 milliseconds.
- Throughput can be scaled horizontally behind an ASGI endpoint (such as FastAPI) with batch query embedding.

### Memory Footprint and Index Caching

- The full 29 contract CUAD corpus generates 6,047 chunks. At 384 dimensions in float32, the FAISS vector index requires 9.3 megabytes of RAM.
- Chunk text metadata and routing indices require an additional 14.5 megabytes.
- Total resident memory per worker process is under 150 megabytes (including PyTorch and sentence transformer weights).
- For large scale enterprise deployments of 100,000 contracts (approximately 20 million chunks), the FAISS index would occupy 31 gigabytes, which can be sharded across nodes or compressed via FAISS IVFPQ (inverted file product quantization) to under 8 gigabytes.

### Dynamic Updates Without Full Re indexing

In commercial legal environments, agreements are executed, revised, or terminated continuously. Rebuilding an entire index upon every document revision is impractical.
- Our modular architecture supports partition level updates: each contract is indexed as an independent chunk partition keyed by document path.
- When an agreement is modified, only its specific chunks are re embedded and updated in the FAISS index using `faiss.IndexIDMap2`, without perturbing other indexed contracts.
- Document routing dictionaries are updated in $O(1)$ time by adding or removing contract title and party aliases.

### Multi Tenant Isolation and Cold Start

- In enterprise environments (such as JioStar content rights management), contract access must enforce strict authorization boundaries (for example, sports rights versus international studio distribution agreements).
- Document routing inherently enforces tenant isolation by restricting search to authorized contract subsets matching user permissions.
- Cold start latency is under 3 seconds: pre built FAISS indexes load from memory mapped disk storage in under 200 milliseconds, and sentence transformer weights load in 2.5 seconds.

## 9. Limitations and Future Directions

### Current Limitations

1. General Domain Embedding Vocabulary: `all-MiniLM-L6-v2` is pretrained on general web text. While it handles semantic similarity effectively, it lacks fine grained understanding of legal distinctions between terms like "indemnification", "subrogation", and "liquidated damages".
2. Multi Paragraph Clause Fragmentation: Fixed 512 character sliding windows inevitably truncate long contractual provisions that span multiple subsections.
3. Overlap with Contractual Recitals: Preambles and recitals frequently mention party definitions and subject matter, attracting dense vector attention over substantive operational clauses.

### Strategic Recommendations

1. Domain Adapted Legal Embedding Fine Tuning: Fine tune sentence transformers using contrastive loss on legal clause pairs (for example, CUAD and EDGAR contract filings) to sharpen distinction between boilerplate and operational terms.
2. Hierarchical Clause Expansion: Store document structural hierarchy (Section $\rightarrow$ Subsection $\rightarrow$ Paragraph). When a chunk is retrieved, dynamically expand the returned character span to encompass the complete parent subsection.
3. Calibrated Cross Encoder with Domain Negatives: Deploy a cross encoder reranker fine tuned specifically on legal hard negatives (distinguishing boilerplate recitals from substantive covenants) to boost true negative abstention beyond 90 percent without sacrificing answerable recall.
