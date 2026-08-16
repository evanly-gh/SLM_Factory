import agent.nodes.iterate as it


def test_decision_prompt_lists_three_strategies():
    prompt = it._build_intervention_prompt_for_test()
    for strategy in ("acquire", "synthesize"):
        assert strategy in prompt
    for gone in (
        "preserve_elite_resample",
        "difficulty_weighted_sampling",
        "source_diversification",
        "query_variant",
        "primary_strategy",
        "support_strategies",
    ):
        assert gone not in prompt, f"stale token {gone!r} still in prompt"
