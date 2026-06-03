"""explore — a code-exploration sub-agent, surfaced as a single tool.

The main dictation agent calls explore("how does auth work?") and gets back a
*map*: the key files (with line numbers), the symbols involved, and a short
synthesis of how they fit together — the same shape as a human explore agent's
report. Internally it runs labern's own tool-use loop with a restricted set of
read-only investigation tools, so one explore call may fan out into several
searches/reads before summarizing.

This keeps the main prompt focused: instead of the top-level agent doing many
searches inline, it delegates "go find where this lives" to explore and gets a
concise answer to weave into the prompt being dictated.
"""

# Tools explore is allowed to use while investigating (read-only; never itself,
# the browser, the shell, or web search — keep it a codebase explorer).
_EXPLORE_TOOLS = ["semantic_search", "file_list", "tree", "read_file",
                  "tree_sitter", "lsp"]

_SYSTEM = """You are a code-exploration agent. Investigate the codebase to answer
WHERE the relevant things live and WHICH parts are relevant to the question.

Use the search / file / code-intelligence tools to locate definitions, call
sites, and the key files. Read enough to be specific and correct. Then produce a
concise MAP for another agent to act on:

- Key files & locations: a short bulleted list of `path:line` with a one-line
  note on what each is.
- Key symbols: the main functions / classes / types involved.
- How it fits: 2-3 sentences on how these pieces connect.

Cite `file:line`. Be concrete, not generic. Do not modify anything."""


def _explore(args, vi=None):
    question = (args.get("question") or args.get("query") or "").strip()
    if not question:
        return "error: 'question' is required"
    if vi is None or not hasattr(vi, "_agent_tool_loop"):
        return "error: explore requires the labern agent runtime"
    from voice_input_tools import TOOLS
    sub_tools = [t for t in _EXPLORE_TOOLS if t in TOOLS]
    if not sub_tools:
        return "error: no investigation tools available"
    try:
        text, _images = vi._agent_tool_loop(question, _SYSTEM, vi.agent_model, sub_tools)
    except Exception as e:
        return f"error: explore failed: {e}"
    return text or "error: explore produced no summary"


TOOLS = {
    "explore": {
        "schema": {
            "name": "explore",
            "description": (
                "Explore the codebase to answer where relevant code lives and which "
                "parts matter, returning a concise map (key files with line numbers, "
                "the symbols involved, and how they fit together). Delegate broad "
                "'where/how is X done' investigation to this instead of searching "
                "inline — it fans out across search and file tools and summarizes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string",
                                 "description": "what to investigate, in natural language"},
                },
                "required": ["question"],
            },
        },
        "run": _explore,
    },
}
