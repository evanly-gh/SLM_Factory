# Qwen3.6-first CoT routing

## Goal

Use the local Qwen3.6 synthesis server as the primary chain-of-thought author for every
generation-family task. Remove Claude and the orchestrator model from CoT generation
entirely. DeepSeek and OpenAI remain cloud fallbacks.

## Routing

For each example that does not already contain `cot_reasoning`:

1. Try local Qwen3.6 through the existing vLLM `generate_fn`.
2. If Qwen is unavailable, raises, or returns empty output, use task-aware fallbacks:
   - `math_reasoning` and math/science generation: DeepSeek, then OpenAI.
   - `code_generation`, code/QA generation, and ordinary generation: OpenAI, then DeepSeek.
3. If every configured backend fails, leave the example unchanged without CoT.

Existing gold CoT is always preserved.

## Components

- Replace `get_teacher_client` with a fallback-list builder that returns ordered
  DeepSeek/OpenAI backends. It must never construct an Anthropic client.
- Extend `annotate_cot` to try one primary local generator followed by the ordered cloud
  fallback list independently for each example.
- Simplify `curate` so it always obtains Qwen first, obtains the task-aware fallback list,
  then makes one `annotate_cot` call.

## Failure behavior and logging

Generation remains non-fatal. Empty responses count as failures and advance to the next
backend. Missing API keys omit that backend from the fallback list. If no backend succeeds,
the original example is retained. Curate logs the primary and fallback order; annotation
logs aggregate success/failure counts rather than one line per example.

## Compatibility

- Preserve the current `annotate_cot` cloud arguments while adding the fallback-list API
  where practical, so focused tests and any external callers do not break unexpectedly.
- Classification and NER paths are unchanged.
- Cheap mode continues to skip CoT annotation.
- No Claude/orchestrator API call is permitted from the CoT path.

## Tests

Add failing tests first for:

1. Qwen succeeds, so neither fallback is called.
2. Qwen fails and math uses DeepSeek before OpenAI.
3. Qwen fails and code/general generation uses OpenAI before DeepSeek.
4. Empty output advances to the next backend.
5. All backends fail, leaving the example unchanged.
6. Gold CoT is preserved without calling any backend.
7. The fallback builder never returns an Anthropic backend.

Run the focused curriculum tests, then the node/cold-start regression suite.
