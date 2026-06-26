# data/curriculum.py
import random
from collections import Counter
from data.eval_set import EvalSet, _infer_pos_label, _infer_neg_label


def build_initial_curriculum(
    train_examples: list[dict],
    eval_set: EvalSet,
    n_total: int = 150,
    gold_fraction: float = 0.65,
    seed: int = 42,
) -> list[dict]:
    """
    Build Dcold = Dgold ∪ Dhard at 65:35 from train_examples.
    Excludes any example that appears in eval_set.
    Applies label balancing (no label > 3x any other).
    Reads task_type from eval_set.task_type internally.
    """
    task_type = eval_set.task_type
    eval_texts = {e["text"] for e in eval_set.all}
    available = [e for e in train_examples if e["text"] not in eval_texts]

    # n_gold is the gold-data portion in a mixed dataset (gold + hard negatives).
    # For the initial curriculum (training data only, no synthetic hard negatives yet),
    # we target n_total examples from the gold pool.
    n_gold = int(n_total * gold_fraction)
    # Use n_total as the selection budget so the initial dataset is large enough.
    selection_budget = n_total

    if task_type == "classification":
        # For classification, balance by label
        by_label: dict[str, list[dict]] = {}
        for ex in available:
            lbl = ex.get("label", "unknown")
            by_label.setdefault(lbl, []).append(ex)

        rng = random.Random(seed)
        for lbl in by_label:
            rng.shuffle(by_label[lbl])

        labels = list(by_label.keys())
        per_label = selection_budget // max(len(labels), 1)
        gold = []
        for lbl in labels:
            gold.extend(by_label[lbl][:per_label])

        # If we still have budget, fill with remaining examples
        selected_texts = {e["text"] for e in gold}
        remainder = [e for e in available if e["text"] not in selected_texts]
        rng.shuffle(remainder)
        gold.extend(remainder[: selection_budget - len(gold)])

    elif task_type in ("NER", "generation"):
        rng = random.Random(seed)
        shuffled = list(available)
        rng.shuffle(shuffled)
        gold = shuffled[:selection_budget]
    else:
        rng = random.Random(seed)
        shuffled = list(available)
        rng.shuffle(shuffled)
        gold = shuffled[:selection_budget]

    return apply_quality_controls(gold, task_type=task_type)


def apply_quality_controls(
    dataset: list[dict],
    task_type: str = "classification",
) -> list[dict]:
    """
    Enforce quality controls from the paper (§2.3). Five constraints:
    1. Label balancing: no label exceeds 3x count of any other (classification)
    2. Context-length matching: remove outliers >3x median length
    3. Entity diversification (NER): cap any entity value at 3 occurrences
    4. Surface-form dedup: remove near-duplicate texts (Jaccard >0.9)
    """
    if not dataset:
        return dataset

    if task_type == "classification":
        clean = [e for e in dataset if "text" in e and "label" in e]

        # 1. Label balancing
        counts = Counter(e["label"] for e in clean)
        if counts:
            min_count = max(min(counts.values()), 1)
            max_allowed = 3 * min_count
            balanced = []
            seen: Counter = Counter()
            for ex in clean:
                if seen[ex["label"]] < max_allowed:
                    balanced.append(ex)
                    seen[ex["label"]] += 1
            clean = balanced

        # 2. Context-length matching: remove length outliers
        clean = _filter_length_outliers(clean)

        # 4. Surface-form dedup
        clean = _dedup_surface_forms(clean)

        return clean

    elif task_type == "NER":
        clean = [e for e in dataset if "text" in e and "entities" in e]

        # 3. Entity diversification: no entity value appears >3 times
        entity_counts: Counter = Counter()
        for ex in clean:
            for ent in ex.get("entities", []):
                entity_counts[ent.get("text", "").lower()] += 1
        over_represented = {k for k, v in entity_counts.items() if v > 3}
        if over_represented:
            result = []
            running: Counter = Counter()
            for ex in clean:
                ent_values = [e.get("text", "").lower() for e in ex.get("entities", [])]
                skip = False
                for v in ent_values:
                    if v in over_represented and running[v] >= 3:
                        skip = True
                        break
                if not skip:
                    result.append(ex)
                    for v in ent_values:
                        running[v] += 1
            clean = result

        clean = _filter_length_outliers(clean)
        return clean

    elif task_type == "generation":
        clean = [
            e for e in dataset
            if ("prompt" in e and "response" in e) or ("text" in e and "label" in e)
        ]
        clean = _filter_length_outliers(clean, key="prompt")
        return clean

    else:
        return dataset


def _filter_length_outliers(
    examples: list[dict], key: str = "text", max_ratio: float = 3.0,
) -> list[dict]:
    """Remove examples whose text length exceeds max_ratio × median. Paper §2.3 item 3."""
    lengths = [len(e.get(key, "")) for e in examples]
    if not lengths:
        return examples
    lengths.sort()
    median = lengths[len(lengths) // 2] or 1
    cutoff = median * max_ratio
    return [e for e in examples if len(e.get(key, "")) <= cutoff]


def _dedup_surface_forms(examples: list[dict], threshold: float = 0.9) -> list[dict]:
    """Remove near-duplicate texts using word-set Jaccard similarity."""
    result = []
    seen_word_sets: list[set] = []
    for ex in examples:
        words = set(ex.get("text", "").lower().split())
        if not words:
            result.append(ex)
            continue
        is_dup = False
        for seen in seen_word_sets[-50:]:
            intersection = len(words & seen)
            union = len(words | seen)
            if union > 0 and intersection / union >= threshold:
                is_dup = True
                break
        if not is_dup:
            result.append(ex)
            seen_word_sets.append(words)
    return result


def synthesize_hard_negatives(
    examples: list[dict],
    n: int,
    anthropic_client,
    task_type: str = "classification",
    targeted_pattern: str = "",
) -> list[dict]:
    """
    Generate hard negatives using the 2-for-1 rule (paper §2.3).

    For each challenging case, returns BOTH the original gold example AND one
    synthetic hard negative — a contrastive pair that teaches the model what
    TO predict and what NOT to predict for similar surface forms.

    Returns up to 2*n examples (n originals + n synthetics).

    Uses Claude API directly (no teacher model needed for phase 1).
    """
    results = []

    if task_type == "classification":
        # Use the full set of examples (all labels) as source material, not just
        # the minority class. Generate a boundary-crossing counterexample for each:
        # given an example with label X, synthesize one that superficially resembles
        # it but belongs to a different label. This covers all failure modes, not just
        # minority-class confusion.
        all_labels = list({e.get("label") for e in examples if e.get("label")})
        candidates = examples[:n]
        pattern_hint = (
            f"\nFocus on this failure pattern: {targeted_pattern}\n"
            if targeted_pattern else ""
        )
        for ex in candidates:
            src_label = ex.get("label", "unknown")
            # Pick a target label different from the source
            target_labels = [l for l in all_labels if l != src_label]
            target_label = target_labels[0] if target_labels else src_label
            prompt = (
                f"Generate a counterexample that looks superficially similar to "
                f"the following example labelled '{src_label}' but has a genuine "
                f"'{target_label}' meaning. Be realistic and subtle.{pattern_hint}\n\n"
                f"Source example: {ex['text']}\n\n"
                f"Respond with only the counterexample text, no explanation."
            )
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            generated_text = response.content[0].text.strip()
            results.append(ex)
            results.append({"text": generated_text, "label": target_label})

    elif task_type == "NER":
        candidates = examples[:n]
        for ex in candidates:
            original_entities = ex.get("entities", [])
            entity_desc = ", ".join(
                f'"{e.get("text", "")}" ({e.get("type", "")})' for e in original_entities[:5]
            ) if original_entities else "unknown entities"
            prompt = (
                f"Rewrite the following passage so that the named entities appear in an "
                f"ambiguous or near-miss context (e.g., 'Apple' as fruit vs company). "
                f"Keep the entity mentions but change their context so the correct entity "
                f"TYPE is different from the original.\n\n"
                f"Original entities: {entity_desc}\n"
                f"Passage: {ex.get('text', '')}\n\n"
                f"Reply with JSON: {{\"text\": \"<rewritten passage>\", "
                f"\"entities\": [{{\"text\": \"<span>\", \"type\": \"<WRONG_TYPE>\"}}]}}\n"
                f"The entities list should contain the spans with INCORRECT entity types "
                f"(the types the model should learn NOT to predict). Reply with JSON only."
            )
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            import json as _json, re as _re
            results.append(ex)
            try:
                match = _re.search(r'\{.*\}', raw, _re.DOTALL)
                parsed = _json.loads(match.group()) if match else {}
                results.append({
                    "text": parsed.get("text", raw),
                    "entities": parsed.get("entities", []),
                })
            except Exception:
                results.append({"text": raw, "entities": []})

    elif task_type == "generation":
        candidates = examples[:n]
        for ex in candidates:
            source_text = ex.get("prompt", ex.get("text", ""))
            gold_answer = ex.get("response", ex.get("label", ex.get("answer", "")))
            prompt = (
                f"Given this question and its correct answer, generate a plausible but "
                f"INCORRECT answer that could trick a language model. The wrong answer "
                f"should sound reasonable but contain a subtle error.\n\n"
                f"Question: {source_text}\n"
                f"Correct answer: {gold_answer}\n\n"
                f"Reply with only the plausible wrong answer, no explanation."
            )
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            wrong_answer = response.content[0].text.strip()
            results.append(ex)
            results.append({"prompt": source_text, "response": wrong_answer})

    return results
