# data/curriculum.py
import random
from collections import Counter
from data.eval_set import EvalSet


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
    Enforce quality controls appropriate for the task type.

    classification:
      - Label balancing: no label exceeds 3x the count of any other.
        Trims the majority class to satisfy the constraint.
      - Removes examples missing 'text' or 'label' fields.

    NER:
      - Removes examples missing 'text' or 'entities' fields.
      - No label balancing (entity distribution is naturally varied).

    generation:
      - Removes examples missing 'prompt' or 'response' fields.
      - No label balancing.
    """
    if not dataset:
        return dataset

    if task_type == "classification":
        # Filter malformed examples
        clean = [e for e in dataset if "text" in e and "label" in e]

        counts = Counter(e["label"] for e in clean)
        if not counts:
            return clean

        min_count = min(counts.values())
        max_allowed = 3 * min_count

        result = []
        seen: Counter = Counter()
        for ex in clean:
            lbl = ex["label"]
            if seen[lbl] < max_allowed:
                result.append(ex)
                seen[lbl] += 1
        return result

    elif task_type == "NER":
        # Filter examples that have both text and entities
        return [e for e in dataset if "text" in e and "entities" in e]

    elif task_type == "generation":
        # Filter examples that have both prompt and response
        # Also accept text/label for datasets that use that convention
        return [
            e for e in dataset
            if ("prompt" in e and "response" in e) or ("text" in e and "label" in e)
        ]

    else:
        # Unknown task type — return as-is
        return dataset


def synthesize_hard_negatives(
    examples: list[dict],
    n: int,
    anthropic_client,
    task_type: str = "classification",
) -> list[dict]:
    """
    Generate n hard negatives using the 2-for-1 rule.

    classification:
      For each boundary spam example, generate a ham counterexample
      with similar surface form but legitimate intent.

    NER:
      For each entity-rich passage, generate a near-miss passage
      where entities are present but in an ambiguous context.

    generation:
      For each clear example, generate an adversarial or ill-posed variant.

    Uses Claude API directly (no teacher model needed for phase 1).
    """
    hard_negatives = []

    if task_type == "classification":
        boundary = [e for e in examples if e.get("label") == "spam"][:n]
        for ex in boundary:
            prompt = (
                f"Generate a legitimate SMS message that looks superficially similar to "
                f"this spam message but has a genuine, non-spam intent.\n\n"
                f"Spam message: {ex['text']}\n\n"
                f"Respond with only the message text, no explanation."
            )
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            generated_text = response.content[0].text.strip()
            hard_negatives.append({"text": generated_text, "label": "ham"})

    elif task_type == "NER":
        candidates = examples[:n]
        for ex in candidates:
            prompt = (
                f"Rewrite the following passage so that the named entities appear in an "
                f"ambiguous or near-miss context (e.g., 'Apple' as fruit vs company).\n\n"
                f"Passage: {ex.get('text', '')}\n\n"
                f"Respond with only the rewritten passage, no explanation."
            )
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            generated_text = response.content[0].text.strip()
            hard_negatives.append({
                "text": generated_text,
                "entities": [],  # gold entities intentionally absent — model must not hallucinate
            })

    elif task_type == "generation":
        candidates = examples[:n]
        for ex in candidates:
            source_text = ex.get("prompt", ex.get("text", ""))
            prompt = (
                f"Generate an adversarial or ill-posed variant of the following prompt "
                f"that could trip up a language model.\n\n"
                f"Original: {source_text}\n\n"
                f"Respond with only the adversarial prompt, no explanation."
            )
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            generated_text = response.content[0].text.strip()
            hard_negatives.append({"prompt": generated_text, "response": None})

    return hard_negatives
