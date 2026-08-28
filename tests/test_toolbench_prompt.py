"""ToolBench's prompt reconstruction must match ToolBench's own, character for character.

WHY THIS FILE EXISTS
    The `toolbench` task takes its train rows from a corpus where ToolBench's system prompt is
    already baked in, and builds the prompt for its eval rows itself, because the ToolEval test
    queries ship as a raw `api_list`. If the rebuild drifts, the model is fine-tuned on one input
    shape and evaluated on another — B250/B290 — and on this task the drift would be invisible: both
    prompts are ~5,000 characters of API schema, so nobody notices a diff by reading a log.

    So the reconstruction is pinned two ways. Against REAL upstream output, using a function dict
    captured verbatim from a ToolBench training row; and by round trip, since the prompt is the only
    thing the loader can read the schemas back out of.
"""
from __future__ import annotations

import pytest

from data.loaders.toolbench_prompt import (
    FINISH_FUNCTION,
    api_function_name,
    api_json_to_openai_json,
    build_functions,
    build_system_prompt,
    build_task_description,
    build_user_prompt,
    change_name,
    process_system_message,
    standardize,
)

# Captured verbatim from `toolllama_G123_dfs_eval.json`, the second declared API of the row whose
# id begins "Step 9: I'm a sports enthusiast". This is upstream's output, not this code's.
GREYHOUND_API = {
    "category_name": "Sports",
    "tool_name": "Greyhound Racing UK",
    "api_name": "Race detail info",
    "api_description": (
        '**Get race detailed info by ID {id_race}.**\n\nYou can get the "id_race" from Results '
        "or Racecards endpoints"
    ),
    "required_parameters": [
        {"name": "id_race", "type": "STRING", "description": "", "default": "53128"},
    ],
    "optional_parameters": [],
}
GREYHOUND_EXPECTED = {
    "name": "race_detail_info_for_greyhound_racing_uk",
    "description": (
        'This is the subfunction for tool "greyhound_racing_uk", you can use this tool.'
        'The description of this function is: "**Get race detailed info by ID {id_race}.**\n\n'
        'You can get the "id_race" from Results or Racecards endpoints"'
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id_race": {"type": "string", "description": "", "example_value": "53128"},
        },
        "required": ["id_race"],
        "optional": [],
    },
}


def test_a_real_api_entry_renders_exactly_as_upstream_rendered_it():
    assert api_json_to_openai_json(GREYHOUND_API, "greyhound_racing_uk") == GREYHOUND_EXPECTED


# --------------------------------------------------------------------------
# standardize / change_name
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [
    ("Greyhound Racing UK", "greyhound_racing_uk"),
    ("Race detail info", "race_detail_info"),
    # Runs of non-alphanumerics collapse to a single underscore, and edges are stripped.
    ("  Weather / Forecast  ", "weather_forecast"),
    ("a--b__c", "a_b_c"),
    ("_leading_and_trailing_", "leading_and_trailing"),
    # A leading digit is not a legal identifier, so upstream prefixes `get_`.
    ("7-day forecast", "get_7_day_forecast"),
    # Entirely punctuation returns empty rather than raising on `string[0]`.
    ("!!!", ""),
])
def test_standardize_matches_upstream(raw, expected):
    assert standardize(raw) == expected


@pytest.mark.parametrize("reserved", ["from", "class", "return", "false", "true", "id", "and"])
def test_a_reserved_parameter_name_is_prefixed(reserved):
    """This is where xLAM's much-queried `is_id` argument comes from.

    Worth pinning because it looks like a typo in the data and is not: a verifier or a reviewer
    "correcting" `is_id` to `id` would produce a call ToolBench's own schema rejects.
    """
    assert change_name(reserved) == f"is_{reserved}"


def test_an_ordinary_parameter_name_is_untouched():
    assert change_name("city") == "city"


def test_a_long_api_name_is_truncated_from_the_FRONT():
    """Upstream's `[-64:]`, not `[:64]`.

    The tool suffix is what disambiguates two same-named APIs across different tools, so keeping
    the head instead would collide them — and the scorer would report the collision as a
    tool-selection error rather than as the naming bug it is.
    """
    name = api_function_name("a" * 80, "some_tool_name")
    assert len(name) == 64
    assert name.endswith("_for_some_tool_name")


# --------------------------------------------------------------------------
# example_value
# --------------------------------------------------------------------------


def test_a_blank_default_produces_no_example_value():
    api = {
        "tool_name": "T", "api_name": "a", "api_description": "",
        "required_parameters": [{"name": "p", "type": "STRING", "description": "", "default": ""}],
        "optional_parameters": [],
    }
    rendered = api_json_to_openai_json(api, "t")
    assert rendered["parameters"]["properties"]["p"] == {"type": "string", "description": ""}


def test_a_zero_default_DOES_produce_an_example_value():
    """Upstream tests `len(str(default)) != 0`, so `0` is a real default and `""` is not.

    Pinned because the obvious simplification — a truthiness test — would silently drop the
    example value for every numeric parameter defaulting to zero.
    """
    api = {
        "tool_name": "T", "api_name": "a", "api_description": "",
        "required_parameters": [{"name": "p", "type": "NUMBER", "description": "", "default": 0}],
        "optional_parameters": [],
    }
    rendered = api_json_to_openai_json(api, "t")
    assert rendered["parameters"]["properties"]["p"]["example_value"] == 0
    assert rendered["parameters"]["properties"]["p"]["type"] == "integer"


def test_optional_parameters_land_in_properties_but_only_in_the_optional_list():
    api = {
        "tool_name": "T", "api_name": "a", "api_description": "",
        "required_parameters": [{"name": "city", "type": "STRING", "description": "",
                                 "default": ""}],
        "optional_parameters": [{"name": "units", "type": "STRING", "description": "",
                                 "default": ""}],
    }
    parameters = api_json_to_openai_json(api, "t")["parameters"]
    assert set(parameters["properties"]) == {"city", "units"}
    assert parameters["required"] == ["city"]
    assert parameters["optional"] == ["units"]


# --------------------------------------------------------------------------
# The assembled prompt
# --------------------------------------------------------------------------


def test_finish_is_declared_last_and_only_once():
    """Every ToolBench prompt ends its API list with `Finish`; it is what makes terminating a path
    legal, and the scorer grades its `return_type` exactly."""
    functions = build_functions([GREYHOUND_API])
    assert [f["name"] for f in functions] == [
        "race_detail_info_for_greyhound_racing_uk", "Finish",
    ]
    assert functions[-1] == FINISH_FUNCTION
    assert functions[-1]["parameters"]["properties"]["return_type"]["enum"] == [
        "give_answer", "give_up_and_restart",
    ]


def test_the_task_description_numbers_each_tool_once_in_first_seen_order():
    api_list = [
        {"tool_name": "Weather API", "api_name": "now"},
        {"tool_name": "Greyhound Racing UK", "api_name": "results"},
        {"tool_name": "Weather API", "api_name": "forecast"},
    ]
    description = build_task_description(
        api_list, {"weather_api": "Weather stuff.", "greyhound_racing_uk": "Dog races."}
    )
    assert "1.weather_api: Weather stuff.\n" in description
    assert "2.greyhound_racing_uk: Dog races.\n" in description
    assert "3." not in description


def test_a_tool_with_no_known_description_says_None_like_upstream():
    """Upstream substitutes the literal string `None` for a blank description, so a tool we have no
    description for produces the same line shape as a tool whose description was empty — rather
    than a differently-shaped line the model has never seen."""
    description = build_task_description([{"tool_name": "Weather API", "api_name": "now"}], {})
    assert "1.weather_api: None\n" in description


def test_the_prompt_states_the_output_format_and_lists_the_apis_as_a_python_repr():
    """`process_system_message` rewrites the step sentence to add the Thought/Action/Action Input
    contract, then appends `str(functions)` — a Python repr with SINGLE quotes, not JSON. The repr
    is not cosmetic: it is what the loader parses the schemas back out of with `ast.literal_eval`.
    """
    prompt = build_system_prompt([GREYHOUND_API], {"greyhound_racing_uk": "Dog races."})
    assert "Your output should follow this format:\nThought:\nAction\nAction Input:\n" in prompt
    assert "Specifically, you have access to the following APIs: [{'name':" in prompt
    assert '"name":' not in prompt.split("following APIs: ")[1]


def test_the_declared_schemas_round_trip_out_of_the_assembled_prompt():
    """The property the loader depends on: whatever is put into a prompt can be read back out.

    If this breaks, `_declared_functions` returns None, every row is dropped as "system prompt
    declares no readable APIs", and the task loads zero rows.
    """
    from data.loaders.toolbench import _declared_functions

    functions = build_functions([GREYHOUND_API])
    prompt = build_system_prompt([GREYHOUND_API], {"greyhound_racing_uk": "Dog races."})
    assert _declared_functions(prompt) == functions


def test_rewriting_a_prompt_that_lost_the_step_sentence_raises_rather_than_silently_skipping():
    """Upstream asserts this. A prompt with no output-format instruction would train the model on a
    contract it was never given, and every row would score as `unparseable_path`."""
    with pytest.raises(ValueError, match="no longer contains the sentence"):
        process_system_message("You are AutoGPT. Do the task.", [])


def test_the_user_turn_wraps_the_query_the_way_toolbench_does():
    assert build_user_prompt("what is the weather in Paris?") == (
        "\nwhat is the weather in Paris?\nBegin!\n"
    )
