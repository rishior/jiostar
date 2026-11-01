"""
Self-improvement loop orchestrator.

Runs the iterative improvement process: evaluate current config, diagnose
failures, propose changes, test proposals, accept or reject based on metrics.
All iterations are logged with full details.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
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


class ImprovementLoop:
    """
    Orchestrates the self-improvement process.
    
    Primary metric: Mean Recall@5 on train set (we want to find all relevant
    passages, and k=5 is a practical number of results for a user).
    
    Guardrail metrics that must not degrade meaningfully:
    1. Mean Precision@5 must not drop by more than 0.05
    2. Abstention rate on unanswerable must stay above 0.5
    3. False abstention rate on answerable must stay below 0.15
    """

    PRIMARY_METRIC = "mean_recall_at_5"
    PRECISION_GUARDRAIL = "mean_precision_at_5"
    MAX_PRECISION_DROP = 0.05
    MIN_UNANSWERABLE_ABSTENTION = 0.5
    MAX_ANSWERABLE_ABSTENTION = 0.15

    def __init__(
        self,
        queries: List[dict],
        unanswerable_queries: Optional[List[dict]] = None,
        max_iterations: int = 10,
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

    def run(self, initial_config: PipelineConfig, verbose: bool = True) -> PipelineConfig:
        """
        Run the full improvement loop starting from the initial configuration.
        
        Returns the best configuration found.
        """
        if verbose:
            print("=" * 60)
            print("SELF-IMPROVEMENT LOOP")
            print("=" * 60)
            print(f"Primary metric: {self.PRIMARY_METRIC}")
            print(f"Guardrails: precision drop < {self.MAX_PRECISION_DROP}, "
                  f"unanswerable abstention > {self.MIN_UNANSWERABLE_ABSTENTION}, "
                  f"answerable abstention < {self.MAX_ANSWERABLE_ABSTENTION}")
            print("=" * 60)

        # Step 1: Evaluate baseline
        if verbose:
            print("\n--- Baseline Evaluation ---")
        
        baseline_pipeline = RetrievalPipeline(initial_config)
        baseline_pipeline.build_index(self.queries)
        
        evaluator = Evaluator(baseline_pipeline)
        baseline_results = evaluator.evaluate(
            self.queries,
            self.unanswerable_queries,
            split="train",
            verbose=verbose,
        )

        self.best_config = initial_config
        self.best_metrics = baseline_results

        # Save baseline config
        initial_config.save(CONFIGS_DIR / "baseline.yaml")

        tried_proposals = set()

        for iteration in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n--- Iteration {iteration} ---")

            # Step 2: Diagnose failures
            answerable_results = [
                qr for qr in baseline_results.query_results if qr.gold_spans
            ]
            unanswerable_results = [
                qr for qr in baseline_results.query_results if not qr.gold_spans
            ]
            diagnosis = diagnose(answerable_results, unanswerable_results)

            if verbose:
                print(f"\nDiagnosis:\n{diagnosis.summary()}")

            # Step 3: Generate proposals
            proposals = generate_proposals(self.best_config, diagnosis, iteration)
            
            # Filter out already-tried proposals
            proposals = [p for p in proposals if p.name not in tried_proposals]

            if not proposals:
                if verbose:
                    print("No new proposals to try. Stopping.")
                break

            # Step 4: Try the top proposal
            proposal = proposals[0]
            tried_proposals.add(proposal.name)

            if verbose:
                print(f"\nProposal: {proposal.name}")
                print(f"Rationale: {proposal.rationale}")

            # Build and evaluate the new pipeline
            try:
                new_pipeline = RetrievalPipeline(proposal.config)
                new_pipeline.build_index(self.queries)

                new_evaluator = Evaluator(new_pipeline)
                new_results = new_evaluator.evaluate(
                    self.queries,
                    self.unanswerable_queries,
                    split="train",
                    verbose=verbose,
                )

                # Step 5: Accept or reject
                accepted, reason = self._should_accept(
                    self.best_metrics, new_results
                )

                log_entry = IterationLog(
                    iteration=iteration,
                    proposal_name=proposal.name,
                    rationale=proposal.rationale,
                    config=proposal.config.to_dict(),
                    metrics_before=self.best_metrics.to_dict(),
                    metrics_after=new_results.to_dict(),
                    diagnosis_summary=diagnosis.summary(),
                    accepted=accepted,
                    reason=reason,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )

                self._log_iteration(log_entry)

                if accepted:
                    if verbose:
                        print(f"ACCEPTED: {reason}")
                    self.best_config = proposal.config
                    self.best_metrics = new_results
                    baseline_results = new_results  # use new results for next diagnosis
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
                )
                self._log_iteration(log_entry)

        # Save final config
        if self.best_config:
            self.best_config.save(CONFIGS_DIR / "final.yaml")
            if verbose:
                print(f"\nBest configuration saved to {CONFIGS_DIR / 'final.yaml'}")

        return self.best_config

    def _should_accept(
        self,
        current: EvaluationResults,
        candidate: EvaluationResults,
    ) -> tuple:
        """
        Decide whether to accept a new configuration.
        
        Returns (accepted: bool, reason: str).
        """
        current_primary = current.mean_recall_at.get(5, 0)
        candidate_primary = candidate.mean_recall_at.get(5, 0)

        current_precision = current.mean_precision_at.get(5, 0)
        candidate_precision = candidate.mean_precision_at.get(5, 0)

        # Check guardrails
        precision_drop = current_precision - candidate_precision
        if precision_drop > self.MAX_PRECISION_DROP:
            return False, (
                f"Precision@5 dropped by {precision_drop:.4f} "
                f"(from {current_precision:.4f} to {candidate_precision:.4f}), "
                f"exceeding max drop of {self.MAX_PRECISION_DROP}"
            )

        if candidate.abstention_rate_answerable > self.MAX_ANSWERABLE_ABSTENTION:
            return False, (
                f"Answerable abstention rate is {candidate.abstention_rate_answerable:.4f}, "
                f"exceeding max of {self.MAX_ANSWERABLE_ABSTENTION}"
            )

        # Check primary metric improvement
        if candidate_primary > current_primary:
            return True, (
                f"Recall@5 improved from {current_primary:.4f} to {candidate_primary:.4f} "
                f"(+{candidate_primary - current_primary:.4f})"
            )
        elif candidate_primary == current_primary:
            # Accept if precision improved at same recall
            if candidate_precision > current_precision:
                return True, (
                    f"Recall@5 unchanged at {current_primary:.4f}, but "
                    f"Precision@5 improved from {current_precision:.4f} to {candidate_precision:.4f}"
                )
            return False, (
                f"Recall@5 unchanged at {current_primary:.4f} with no precision improvement"
            )
        else:
            return False, (
                f"Recall@5 decreased from {current_primary:.4f} to {candidate_primary:.4f} "
                f"({candidate_primary - current_primary:.4f})"
            )

    def _log_iteration(self, log_entry: IterationLog):
        """Append an iteration log entry to the experiment log file."""
        self.iteration_logs.append(log_entry)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(log_entry)) + "\n")
