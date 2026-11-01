"""
Self-improvement loop orchestrator.

Runs the iterative improvement process: evaluate current config, diagnose
failures, propose changes, test proposals, accept or reject based on metrics.
All iterations are logged with full details to logs/experiment_log.jsonl.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from src.config import PipelineConfig, LOGS_DIR, CONFIGS_DIR
from src.retrieval.pipeline import RetrievalPipeline
from src.evaluation.evaluator import Evaluator, EvaluationResults
from src.improvement.diagnosis import diagnose, DiagnosisResult
from src.improvement.proposals import generate_proposals, Proposal


@dataclass
class IterationLog:
    """Log entry for a single improvement iteration."""
    iteration: int
    proposal_name: str
    rationale: str
    config: dict
    metrics_before: dict
    metrics_after: dict
    diagnosis_summary: str
    accepted: bool
    reason: str
    timestamp: str = ""
    proposed_by: str = "diagnosis"
    run_label: str = "v2"


class ImprovementLoop:
    """
    Orchestrates the self-improvement process.

    Primary metric: Mean F1@5 on the training split.

    Guardrail metrics that must not degrade meaningfully:
    1. Mean Precision@5 must not drop by more than 10 percent relative
    2. Abstention rate on answerable queries must stay below 0.15
    3. In-corpus unanswerable abstention rate must not decrease
    4. Minimum primary metric gain of 0.005 required to accept
    """

    PRIMARY_METRIC = "mean_f1_at_5"
    MAX_RELATIVE_PRECISION_DROP = 0.10
    MAX_ANSWERABLE_ABSTENTION = 0.15
    MIN_F1_GAIN = 0.005

    def __init__(
        self,
        queries: List[dict],
        unanswerable_queries: Optional[List[dict]] = None,
        max_iterations: int = 6,
        log_path: Optional[Path] = None,
    ):
        self.queries = queries
        self.unanswerable_queries = unanswerable_queries or []
        self.max_iterations = max_iterations
        self.log_path = log_path or (LOGS_DIR / "experiment_log.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.iteration_logs: List[IterationLog] = []
        self.best_config: Optional[PipelineConfig] = None
        self.best_metrics: Optional[EvaluationResults] = None

    def _get_existing_iterations(self) -> List[dict]:
        """Read existing entries from experiment_log.jsonl if present."""
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        return entries

    def run(self, initial_config: PipelineConfig, verbose: bool = True) -> PipelineConfig:
        """
        Run the iterative improvement loop starting from initial_config.
        Evaluates strictly on the training split.
        Appends new iteration logs to experiment_log.jsonl.
        Saves the best configuration to configs/final_v2.yaml.
        """
        existing_entries = self._get_existing_iterations()
        start_iteration = 1
        if existing_entries:
            max_existing = max(e.get("iteration", 0) for e in existing_entries)
            start_iteration = max_existing + 1

        if verbose:
            print("=" * 60)
            print("SELF IMPROVEMENT LOOP (v2)")
            print("=" * 60)
            print(f"Starting iteration: {start_iteration}")
            print(f"Max new iterations: {self.max_iterations}")
            print(f"Primary metric: {self.PRIMARY_METRIC}")
            print(f"Guardrails: relative precision drop < 10%, "
                  f"answerable abstention < {self.MAX_ANSWERABLE_ABSTENTION}, "
                  f"in corpus abstention non decreasing")
            print("=" * 60)

        # Evaluate the starting configuration on train
        if verbose:
            print(f"\n--- Initial Configuration Evaluation ({initial_config.name}) ---")

        current_pipeline = RetrievalPipeline(initial_config)
        current_pipeline.build_index(self.queries)

        evaluator = Evaluator(current_pipeline)
        current_results = evaluator.evaluate(
            self.queries,
            self.unanswerable_queries,
            split="train",
            verbose=verbose,
        )

        self.best_config = initial_config
        self.best_metrics = current_results

        tried_proposals = set()
        # Collect previously tried proposal names from history
        for e in existing_entries:
            p_name = e.get("proposal_name")
            if p_name:
                tried_proposals.add(p_name)

        end_iteration = start_iteration + self.max_iterations

        for iteration in range(start_iteration, end_iteration):
            if verbose:
                print(f"\n--- Iteration {iteration} ---")

            # Diagnose current failures
            answerable_results = [
                qr for qr in self.best_metrics.query_results if qr.gold_spans
            ]
            unanswerable_results = [
                qr for qr in self.best_metrics.query_results if not qr.gold_spans
            ]
            diagnosis = diagnose(answerable_results, unanswerable_results)

            if verbose:
                print(f"\nDiagnosis:\n{diagnosis.summary()}")

            # Generate proposals
            proposals = generate_proposals(self.best_config, diagnosis, iteration)
            proposals = [p for p in proposals if p.name not in tried_proposals]

            if not proposals:
                if verbose:
                    print("No further proposals to evaluate; stopping loop.")
                break

            proposal = proposals[0]
            tried_proposals.add(proposal.name)

            if verbose:
                print(f"\nSelected Proposal: {proposal.name}")
                print(f"Proposed by: {proposal.proposed_by}")
                print(f"Rationale: {proposal.rationale}")

            try:
                candidate_pipeline = RetrievalPipeline(proposal.config)
                candidate_pipeline.build_index(self.queries)

                candidate_evaluator = Evaluator(candidate_pipeline)
                candidate_results = candidate_evaluator.evaluate(
                    self.queries,
                    self.unanswerable_queries,
                    split="train",
                    verbose=verbose,
                )

                accepted, reason = self._should_accept(
                    self.best_metrics, candidate_results
                )

                log_entry = IterationLog(
                    iteration=iteration,
                    proposal_name=proposal.name,
                    rationale=proposal.rationale,
                    config=proposal.config.to_dict(),
                    metrics_before=self.best_metrics.to_dict(),
                    metrics_after=candidate_results.to_dict(),
                    diagnosis_summary=diagnosis.summary(),
                    accepted=accepted,
                    reason=reason,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    proposed_by=proposal.proposed_by,
                    run_label="v2",
                )

                self._log_iteration(log_entry)

                if accepted:
                    if verbose:
                        print(f"ACCEPTED: {reason}")
                    self.best_config = proposal.config
                    self.best_metrics = candidate_results
                    # Save intermediate checkpoint
                    self.best_config.save(CONFIGS_DIR / "final_v2.yaml")
                else:
                    if verbose:
                        print(f"REJECTED: {reason}")

            except Exception as e:
                if verbose:
                    print(f"ERROR evaluating proposal: {e}")
                log_entry = IterationLog(
                    iteration=iteration,
                    proposal_name=proposal.name,
                    rationale=proposal.rationale,
                    config=proposal.config.to_dict(),
                    metrics_before=self.best_metrics.to_dict(),
                    metrics_after={},
                    diagnosis_summary=diagnosis.summary(),
                    accepted=False,
                    reason=f"Error during evaluation: {str(e)}",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    proposed_by=proposal.proposed_by,
                    run_label="v2",
                )
                self._log_iteration(log_entry)

        # Save final v2 config
        if self.best_config:
            self.best_config.save(CONFIGS_DIR / "final_v2.yaml")
            if verbose:
                print(f"\nBest v2 configuration saved to {CONFIGS_DIR / 'final_v2.yaml'}")

        return self.best_config

    def _should_accept(
        self,
        current: EvaluationResults,
        candidate: EvaluationResults,
    ) -> Tuple[bool, str]:
        """
        Decide whether to accept candidate configuration based on F1@5 and guardrails.
        """
        current_f1 = current.mean_f1_at.get(5, 0.0)
        candidate_f1 = candidate.mean_f1_at.get(5, 0.0)

        current_precision = current.mean_precision_at.get(5, 0.0)
        candidate_precision = candidate.mean_precision_at.get(5, 0.0)

        # Guardrail 1: Answerable false abstention must be below 0.15
        if candidate.abstention_rate_answerable > self.MAX_ANSWERABLE_ABSTENTION:
            return False, (
                f"Answerable abstention rate is {candidate.abstention_rate_answerable:.4f}, "
                f"exceeding maximum permitted {self.MAX_ANSWERABLE_ABSTENTION}"
            )

        # Guardrail 2: Relative precision drop must not exceed 10%
        if current_precision > 0:
            rel_precision_drop = (current_precision - candidate_precision) / current_precision
            if rel_precision_drop > self.MAX_RELATIVE_PRECISION_DROP:
                return False, (
                    f"Precision@5 dropped by {rel_precision_drop * 100:.1f}% relative "
                    f"(from {current_precision:.4f} to {candidate_precision:.4f}), "
                    f"exceeding max allowable drop of 10%"
                )

        # Guardrail 3: In corpus unanswerable abstention must not decrease
        current_in_corpus = current.abstention_rate_unanswerable_in_corpus
        candidate_in_corpus = candidate.abstention_rate_unanswerable_in_corpus
        if candidate_in_corpus < current_in_corpus - 1e-4:
            return False, (
                f"In corpus unanswerable abstention decreased from {current_in_corpus:.4f} "
                f"to {candidate_in_corpus:.4f}"
            )

        # Primary metric evaluation
        f1_gain = candidate_f1 - current_f1

        # Case A: F1@5 improved by at least MIN_F1_GAIN
        if f1_gain >= self.MIN_F1_GAIN:
            return True, (
                f"F1@5 improved from {current_f1:.4f} to {candidate_f1:.4f} (+{f1_gain:.4f})"
            )

        # Case B: Significant in corpus abstention improvement with stable F1
        in_corpus_gain = candidate_in_corpus - current_in_corpus
        if in_corpus_gain >= 0.05 and candidate_f1 >= current_f1 - 0.005:
            return True, (
                f"In corpus unanswerable abstention improved from {current_in_corpus:.4f} "
                f"to {candidate_in_corpus:.4f} (+{in_corpus_gain:.4f}) with stable F1@5 ({candidate_f1:.4f})"
            )

        # Case C: Recall or precision improved with equal F1
        if candidate_f1 >= current_f1 and candidate_precision > current_precision:
            return True, (
                f"F1@5 maintained at {candidate_f1:.4f} while Precision@5 improved "
                f"from {current_precision:.4f} to {candidate_precision:.4f}"
            )

        return False, (
            f"F1@5 gain {f1_gain:.4f} is below minimum threshold of {self.MIN_F1_GAIN} "
            f"(current {current_f1:.4f}, candidate {candidate_f1:.4f})"
        )

    def _log_iteration(self, log_entry: IterationLog):
        """Append an iteration log entry to the experiment log file."""
        self.iteration_logs.append(log_entry)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(log_entry)) + "\n")
