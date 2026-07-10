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
    from config.config import ORCHESTRATOR_MODEL, ANTHROPIC_API_KEY

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    # Phase 1 sub-agents have NO tools, so they cannot write files. The prompt is
    # honest about this: return the structured result inline; the caller persists it
    # to `output_file`. (An earlier version told the model to "write the file", which
    # it could never do — the caller then silently fell back to the response text.)
    system = (
        "You are a focused sub-agent in an agentic fine-tuning pipeline. You have NO "
        "file-system or tool access, so you cannot write files. Complete the task and "
        "return your complete, structured result as the body of your reply — it will be "
        "captured and saved by the calling agent. Be concise and return only the result "
        "(no preamble)."
    )
    messages = [{"role": "user", "content": task_description}]

    response = client.messages.create(
        model=ORCHESTRATOR_MODEL,
        max_tokens=4096,
        system=system,
        messages=messages,
    )

    result_text = next((b.text for b in response.content if b.type == "text"), "")
    # Persist the sub-agent's result on its behalf so callers can rely on output_file.
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result_text)
    except OSError:
        pass
    return result_text
