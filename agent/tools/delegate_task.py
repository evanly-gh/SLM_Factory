# agent/tools/delegate_task.py
"""
Sub-agent spawning via delegate_task.
Sub-agents share the filesystem. Main agent reads their output files.
Not a named tool — called programmatically by the orchestrator.
"""
import anthropic


def delegate_task(task_description: str, output_file: str) -> str:
    """
    Spawn a sub-agent to work on task_description.
    Sub-agent writes its result to output_file on disk.
    Main agent reads output_file — never gets raw sub-agent context.
    Returns the contents of output_file when complete.
    """
    from config import ORCHESTRATOR_MODEL, MAX_TURNS_SUBAGENT, ANTHROPIC_API_KEY

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system = (
        "You are a sub-agent in an agentic fine-tuning pipeline. "
        f"Complete the task and write your structured output to: {output_file}\n"
        "Be concise. Write the file before responding."
    )
    messages = [{"role": "user", "content": task_description}]

    # Simple single-turn sub-agent for Phase 1
    # Phase 2 can extend to multi-turn with tool use
    response = client.messages.create(
        model=ORCHESTRATOR_MODEL,
        max_tokens=4096,
        system=system,
        messages=messages,
    )

    # Read output file written by sub-agent (if it did so via tool calls in extended use)
    try:
        with open(output_file, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return response.content[0].text
