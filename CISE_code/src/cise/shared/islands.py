from __future__ import annotations

import dataclasses
import random
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


Signature = Tuple[float, ...]
ScoresPerTest = Mapping[str, float]


def candidate_success(candidate: Mapping[str, Any]) -> bool:

    if candidate.get('abstained', False):
        return False
    return bool(candidate.get('interval_pass', candidate.get('proxy_pass', False)))


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    values = [scores_per_test[k] for k in scores_per_test.keys()]
    return float(sum(values) / len(values)) if values else float('-inf')


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.all(np.isfinite(logits)):
        logits = np.array([0.0])
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)
    if logits.size == 0:
        logits = np.array([0.0], dtype=np.float32)
    probs = np.exp(logits / max(temperature, 1e-6))
    probs_sum = probs.sum()
    if probs_sum <= 0:
        probs = np.ones_like(probs)
        probs_sum = probs.sum()
    probs = probs / probs_sum

    idx = int(np.argmax(probs))
    probs[idx] = 1.0 - (probs.sum() - probs[idx])
    return probs


@dataclasses.dataclass(frozen=True)
class Prompt:
    memory_success: List[Dict[str, Any]]
    memory_failure: List[Dict[str, Any]]
    island_id: int


class Cluster:
    def __init__(self, score: float, candidate: Dict[str, Any]):
        self._score = float(score)
        self._candidates: List[Dict[str, Any]] = [candidate]

    @property
    def score(self) -> float:
        return self._score

    def register(self, candidate: Dict[str, Any]) -> None:
        self._candidates.append(candidate)

    def sample(self) -> Dict[str, Any]:
        return random.choice(self._candidates)


class Island:
    def __init__(self, functions_per_prompt: int, temp_init: float, temp_period: int, memory_max_items: int = 50, memory_top_k_success: int = 10, memory_bottom_k_failure: int = 10):
        self._clusters: Dict[Signature, Cluster] = {}
        self._num_items: int = 0
        self._functions_per_prompt = functions_per_prompt
        self._temp_init = temp_init
        self._temp_period = max(1, temp_period)
        self._memory_max_items = memory_max_items
        self._memory_top_k_success = memory_top_k_success
        self._memory_bottom_k_failure = memory_bottom_k_failure


        self._unique_candidates: Dict[str, Dict[str, Any]] = {}
        self._iteration_history: List[Dict[str, Any]] = []
        self._refresh_interval = 50
        self._last_refresh_iteration = 0

    def register_candidate(self, candidate: Dict[str, Any], scores_per_test: ScoresPerTest, iteration: int = 0) -> None:
        if candidate.get('abstained', False):
            return

        formula = candidate.get('formula', '')
        compound = candidate.get('compound', '')
        candidate_key = formula or compound or str(hash(str(candidate)))


        if candidate_key not in self._unique_candidates:
            self._unique_candidates[candidate_key] = candidate
        else:

            current_score = self._unique_candidates[candidate_key].get('score', float('-inf'))
            new_score = candidate.get('score', float('-inf'))
            if new_score > current_score:
                self._unique_candidates[candidate_key] = candidate


        candidate_with_iteration = candidate.copy()
        candidate_with_iteration['_iteration'] = iteration
        self._iteration_history.append(candidate_with_iteration)


        sig = _get_signature(scores_per_test)
        if sig not in self._clusters:
            self._clusters[sig] = Cluster(_reduce_score(scores_per_test), candidate)
        else:
            self._clusters[sig].register(candidate)
        self._num_items += 1

    def get_prompt_memories(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

        all_items: List[Tuple[float, Dict[str, Any]]] = []
        for cl in self._clusters.values():

            for c in cl._candidates:
                candidate_score = float(c.get('score', 0.0))
                all_items.append((candidate_score, c))

        if not all_items:
            return [], []


        all_items.sort(key=lambda x: x[0], reverse=True)

        successes = [c for _, c in all_items if candidate_success(c)]
        failures = [c for _, c in all_items if not candidate_success(c)]


        successes = successes[:min(self._memory_top_k_success, len(successes))]
        failures = failures[:min(self._memory_bottom_k_failure, len(failures))]

        return successes, failures

    def refresh_memory_if_needed(self, current_iteration: int) -> None:

        if current_iteration - self._last_refresh_iteration >= self._refresh_interval:
            print(f"[Agent] Refreshing island memory at iteration {current_iteration}")


            cutoff_iteration = current_iteration - self._refresh_interval
            self._iteration_history = [
                c for c in self._iteration_history
                if c.get('_iteration', 0) > cutoff_iteration
            ]


            self._unique_candidates = {}
            for candidate in self._iteration_history:
                formula = candidate.get('formula', '')
                compound = candidate.get('compound', '')
                candidate_key = formula or compound or str(hash(str(candidate)))

                if candidate_key not in self._unique_candidates:
                    self._unique_candidates[candidate_key] = candidate
                else:
                    current_score = self._unique_candidates[candidate_key].get('score', float('-inf'))
                    new_score = candidate.get('score', float('-inf'))
                    if new_score > current_score:
                        self._unique_candidates[candidate_key] = candidate

            self._last_refresh_iteration = current_iteration
            print(f"[Agent] Island memory refreshed: {len(self._unique_candidates)} unique candidates remaining")

    def get_unique_examples(self, max_examples: int = 20, rank_candidates=None) -> Dict[str, List[Dict[str, Any]]]:


        all_candidates = list(self._unique_candidates.values())
        if rank_candidates is not None:


            ranked = rank_candidates(self._iteration_history)
            unique = {}
            for candidate in ranked:
                key = candidate.get('formula') or candidate.get('compound') or str(hash(str(candidate)))
                if key not in unique:
                    unique[key] = candidate
            all_candidates = list(unique.values())

        if not all_candidates:
            return {'success': [], 'failure': []}


        all_candidates.sort(key=lambda c: c.get('score', 0), reverse=True)


        successes = [c for c in all_candidates if candidate_success(c)]
        failures = [c for c in all_candidates if not candidate_success(c)]


        successes = successes[:max_examples]
        failures = failures[:max_examples]

        return {'success': successes, 'failure': failures}

    def is_duplicate(self, candidate: Dict[str, Any]) -> bool:

        formula = candidate.get('formula', '')
        compound = candidate.get('compound', '')
        candidate_key = formula or compound or str(hash(str(candidate)))
        return candidate_key in self._unique_candidates

    def sample_clusters_indices(self) -> List[int]:
        signatures = list(self._clusters.keys())
        if not signatures:
            return []
        scores = np.array([self._clusters[s].score for s in signatures], dtype=np.float32)

        temperature = 0.8
        probs = _softmax(scores, max(temperature, 1e-6))
        k = min(len(signatures), self._functions_per_prompt)
        idx = np.random.choice(len(signatures), size=k, p=probs, replace=False)
        return list(idx)


class ExperienceBuffer:
    def __init__(self, num_islands: int, functions_per_prompt: int = 4, temp_init: float = 1.0, temp_period: int = 10, reset_period_seconds: int = 120, max_items_per_island: int = 50, top_k_success: int = 10, bottom_k_failure: int = 10):
        self._islands: List[Island] = [Island(functions_per_prompt, temp_init, temp_period, max_items_per_island, top_k_success, bottom_k_failure) for _ in range(max(1, num_islands))]
        self._best_score_per_island: List[float] = [-float('inf')] * len(self._islands)
        self._best_candidate_per_island: List[Dict[str, Any] | None] = [None] * len(self._islands)
        self._best_scores_per_test_per_island: List[Dict[str, float] | None] = [None] * len(self._islands)
        self._last_reset_time: float = time.time()
        self._reset_period_seconds = reset_period_seconds
        self._max_items_per_island = max_items_per_island
        self._top_k_success = top_k_success
        self._bottom_k_failure = bottom_k_failure

    def register(self, island_id: int, candidate: Dict[str, Any], scores_per_test: ScoresPerTest, iteration: int = 0) -> None:
        if candidate.get('abstained', False):
            return
        self._islands[island_id].register_candidate(candidate, scores_per_test, iteration)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_candidate_per_island[island_id] = candidate
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
        self._maybe_reset()

    def seed_all(self, candidates: List[Dict[str, Any]], score_key: str = 'score') -> None:
        candidates = [c for c in candidates if not c.get('abstained', False)]
        if not candidates:
            return
        print(f"[Agent] SEEDING {len(candidates)} candidates to ALL {len(self._islands)} islands:")

        for c in candidates:
            score = float(c.get(score_key, 0.0))
            formula = c.get('formula', 'Unknown')

            scores_per_test = {'total': score}

            if 'predictions' in c:
                for prop, value in c['predictions'].items():
                    if isinstance(value, (int, float)):
                        scores_per_test[prop] = float(value)


            for island_id in range(len(self._islands)):
                self.register(island_id, c, scores_per_test)
            print(f"  - Seeded {formula} (score: {score:.3f}) to ALL islands")

    def get_prompt(self, iteration: int = 0) -> Prompt:
        island_id = int(np.random.randint(len(self._islands)))

        self._islands[island_id].refresh_memory_if_needed(iteration)

        self._prune_island(island_id)
        successes, failures = self._islands[island_id].get_prompt_memories()
        return Prompt(successes, failures, island_id)

    def get_unique_examples_for_evolution(self, island_id: int, max_examples: int = 20, rank_candidates=None) -> Dict[str, List[Dict[str, Any]]]:

        return self._islands[island_id].get_unique_examples(max_examples, rank_candidates=rank_candidates)

    def is_duplicate_candidate(self, island_id: int, candidate: Dict[str, Any]) -> bool:

        return self._islands[island_id].is_duplicate(candidate)

    def _prune_island(self, island_id: int) -> None:
        island = self._islands[island_id]

        items: List[Tuple[float, Dict[str, Any]]] = []
        for cl in island._clusters.values():
            items.extend([(cl.score, c) for c in cl._candidates])
        if len(items) <= self._max_items_per_island:
            return
        items.sort(key=lambda x: x[0], reverse=True)
        keep: List[Dict[str, Any]] = [c for _, c in items[: self._top_k_success]] + [c for _, c in items[-self._bottom_k_failure :]]

        island._clusters = {}
        island._num_items = 0
        for c in keep:
            score = float(c.get('score', 0.0))
            island.register_candidate(c, {'total': score})

    def _maybe_reset(self) -> None:
        if time.time() - self._last_reset_time < self._reset_period_seconds:
            return
        self._last_reset_time = time.time()
        scores = np.array(self._best_score_per_island, dtype=np.float32)
        order = np.argsort(scores)
        half = len(order) // 2
        to_reset = order[:half]
        keep = order[half:]
        if keep.size == 0:
            return
        founder = int(np.random.choice(keep))

        print(f"[Agent] ISLAND RESET triggered!")
        print(f"  - Islands to reset: {to_reset.tolist()}")
        print(f"  - Islands to keep: {keep.tolist()}")
        print(f"  - Founder island: {founder} (score: {self._best_score_per_island[founder]:.3f})")


        for idx in to_reset:
            old_score = self._best_score_per_island[idx]
            self._islands[idx] = Island(
                self._islands[founder]._functions_per_prompt,
                self._islands[founder]._temp_init,
                self._islands[founder]._temp_period,
                self._max_items_per_island,
                self._top_k_success,
                self._bottom_k_failure
            )
            self._best_score_per_island[idx] = -float('inf')
            self._best_candidate_per_island[idx] = None
            self._best_scores_per_test_per_island[idx] = None


            if self._best_candidate_per_island[founder] is not None:
                founder_candidate = self._best_candidate_per_island[founder]
                founder_scores = self._best_scores_per_test_per_island[founder]
                self.register(idx, founder_candidate, founder_scores)
                print(f"  - Reset island {idx} (old best score: {old_score:.3f}) and seeded with founder {founder}'s best candidate")
            else:
                print(f"  - Reset island {idx} (old best score: {old_score:.3f}) but no founder candidate available")
