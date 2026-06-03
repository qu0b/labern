"""tree_sitter tool: extract a compact symbol/structure outline of a source file.

A tool is an entry in TOOLS: an Anthropic tool `schema` (name/description/
input_schema) plus a `run(args, vi)` callable returning a string the model reads
back as a tool_result. `vi` is the VoiceInput instance — tools read config off it
(e.g. the project root). Keep results small; the model re-reads them every turn.

This one parses a file with tree-sitter (via the pre-installed
`tree-sitter-language-pack`), walks the syntax tree, and emits a nested outline of
definitions: `kind name (L<line>)`, one per line, indented by nesting depth.

Note on the binding: the parser shipped by tree-sitter-language-pack exposes a
method-based Node API (`node.kind()`, `node.child(i)`, `node.start_position()`,
`node.child_by_field_name(...)`, `node.start_byte()`/`end_byte()`) and has no
`node.text`, so symbol names are recovered by slicing the source bytes.
"""

import os

# Map file extension -> tree-sitter language name understood by get_parser().
_EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".rb": "ruby",
}

# Per-language: tree-sitter node kind -> friendly outline kind. Only kinds we want
# to surface appear here; everything else is descended into but not printed.
_KINDS = {
    "python": {
        "function_definition": "function",  # refined to "method" inside a class
        "class_definition": "class",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "interface",
        "impl_item": "impl",
        "mod_item": "module",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "type",  # refined via inner type_spec
    },
    "java": {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "method_declaration": "method",
        "constructor_declaration": "method",
    },
    "ruby": {
        "class": "class",
        "module": "module",
        "method": "method",
        "singleton_method": "method",
    },
    # JavaScript / TypeScript / TSX share one mapping.
    "_jsts": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "lexical_declaration": "function",  # only when it binds an arrow/function
        "variable_declaration": "function",
    },
    # C / C++ share one mapping.
    "_cfamily": {
        "function_definition": "function",
        "struct_specifier": "struct",
        "class_specifier": "class",
        "enum_specifier": "enum",
        "union_specifier": "struct",
        "namespace_definition": "module",
        "field_declaration": "method",  # a method declaration inside a class body
        "type_definition": "type",
    },
}
for _l in ("javascript", "typescript", "tsx"):
    _KINDS[_l] = _KINDS["_jsts"]
for _l in ("c", "cpp"):
    _KINDS[_l] = _KINDS["_cfamily"]

_MAX_BYTES = 8000


def _slice(src_bytes, node):
    """Decoded source text spanned by a node (replacing undecodable bytes)."""
    if node is None:
        return None
    return src_bytes[node.start_byte():node.end_byte()].decode("utf8", "replace")


def _identifier_from_declarator(src_bytes, node):
    """C/C++: drill through nested `declarator` children to the leaf identifier."""
    seen = 0
    cur = node
    while cur is not None and seen < 8:
        seen += 1
        kind = cur.kind()
        if kind in ("identifier", "field_identifier", "type_identifier",
                    "qualified_identifier", "destructor_name", "operator_name"):
            return _slice(src_bytes, cur)
        nxt = cur.child_by_field_name("declarator")
        if nxt is None:
            # Fall back to the first identifier-ish named child.
            for i in range(cur.child_count()):
                c = cur.child(i)
                if c.kind() in ("identifier", "field_identifier"):
                    return _slice(src_bytes, c)
            return None
        cur = nxt
    return None


def _node_name(src_bytes, lang, node):
    """Best-effort symbol name for a definition node, or None."""
    kind = node.kind()
    # Direct `name` field covers most languages (py, rust, go type_spec, java,
    # ruby, js/ts function/class/interface/type, c/cpp struct/class/enum/namespace).
    nm = node.child_by_field_name("name")
    if nm is not None:
        return _slice(src_bytes, nm)
    if lang == "rust" and kind == "impl_item":
        # impl has no name; report the type it implements (+ trait if present).
        ty = node.child_by_field_name("type")
        trait = node.child_by_field_name("trait")
        t = _slice(src_bytes, ty)
        tr = _slice(src_bytes, trait)
        if t and tr:
            return f"{tr} for {t}"
        return t
    if lang in ("c", "cpp"):
        # function_definition / field_declaration carry the name under declarator.
        decl = node.child_by_field_name("declarator")
        if decl is not None:
            return _identifier_from_declarator(src_bytes, decl)
    return None


def _refine_kind(lang, node, friendly, in_class):
    """Adjust the friendly kind using context the static map can't capture."""
    kind = node.kind()
    if lang == "python" and kind == "function_definition" and in_class:
        return "method"
    if lang == "go" and kind == "type_declaration":
        # Classify by the wrapped type: struct / interface / plain type alias.
        for i in range(node.child_count()):
            spec = node.child(i)
            if spec.kind() != "type_spec":
                continue
            ty = spec.child_by_field_name("type")
            if ty is not None:
                tk = ty.kind()
                if tk == "struct_type":
                    return "struct"
                if tk == "interface_type":
                    return "interface"
        return "type"
    return friendly


def _go_type_name(src_bytes, node):
    """Name for a Go `type_declaration` lives on its inner `type_spec`."""
    for i in range(node.child_count()):
        spec = node.child(i)
        if spec.kind() == "type_spec":
            nm = spec.child_by_field_name("name")
            if nm is not None:
                return _slice(src_bytes, nm)
    return None


def _has_function_declarator(node):
    """C/C++: True if a declarator chain contains a `function_declarator`
    (distinguishes a method declaration from a plain data field)."""
    seen = 0
    cur = node
    while cur is not None and seen < 8:
        seen += 1
        if cur.kind() == "function_declarator":
            return True
        cur = cur.child_by_field_name("declarator")
    return False


def _jsts_binding(src_bytes, node):
    """For a (lexical|variable)_declaration, return (name, has_fn) if it binds a
    function/arrow expression, else (None, False) so we skip plain consts."""
    for i in range(node.child_count()):
        dec = node.child(i)
        if dec.kind() != "variable_declarator":
            continue
        val = dec.child_by_field_name("value")
        if val is not None and val.kind() in ("arrow_function", "function",
                                              "function_expression",
                                              "generator_function"):
            nm = dec.child_by_field_name("name")
            return (_slice(src_bytes, nm), True)
    return (None, False)


def _is_class_kind(lang, friendly, kind):
    """Does this node introduce a class-like scope (so children are 'methods')?"""
    if friendly in ("class", "struct", "interface", "impl"):
        return True
    if lang == "go" and friendly in ("struct", "interface"):
        return True
    return False


def _outline(src_bytes, lang):
    """Return the list of outline lines for a parsed source buffer."""
    from tree_sitter_language_pack import get_parser

    tree = get_parser(lang).parse(src_bytes.decode("utf8", "replace"))
    kindmap = _KINDS.get(lang, {})
    lines = []

    def walk(node, depth, in_class):
        kind = node.kind()
        friendly = kindmap.get(kind) if node.is_named() else None
        emitted = False
        if friendly is not None:
            if lang in ("c", "cpp") and kind == "field_declaration" \
                    and not _has_function_declarator(node):
                friendly = None  # plain data member, not a method
        if friendly is not None:
            name = None
            if lang == "go" and kind == "type_declaration":
                name = _go_type_name(src_bytes, node)
            elif lang in ("javascript", "typescript", "tsx") and kind in (
                    "lexical_declaration", "variable_declaration"):
                name, is_fn = _jsts_binding(src_bytes, node)
                if not is_fn:
                    friendly = None  # plain const/let/var, not a definition
            else:
                name = _node_name(src_bytes, lang, node)

            if friendly is not None:
                friendly = _refine_kind(lang, node, friendly, in_class)
                if name is None:
                    name = "<anonymous>"
                line = node.start_position().row + 1
                lines.append(f"{'  ' * depth}{friendly} {name} (L{line})")
                emitted = True

        child_depth = depth + 1 if emitted else depth
        child_in_class = in_class
        if emitted:
            child_in_class = _is_class_kind(lang, friendly, kind)
        for i in range(node.child_count()):
            walk(node.child(i), child_depth, child_in_class)

    walk(tree.root_node(), 0, False)
    return lines


def _run(args, vi=None):
    """Parse `path` and return a compact nested outline of its definitions."""
    try:
        path = (args.get("path") or "").strip()
        if not path:
            return "error: 'path' is required"
        root = (getattr(vi, "tools_config", {}) or {}).get("root") or os.getcwd()
        if not os.path.isabs(path):
            path = os.path.join(root, path)
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            return f"error: not a file: {path}"

        ext = os.path.splitext(path)[1].lower()
        lang = _EXT_LANG.get(ext)
        if lang is None:
            return f"error: unsupported language: {ext or '(no extension)'}"

        try:
            from tree_sitter_language_pack import get_parser  # noqa: F401
        except ImportError:
            return "error: tree-sitter-language-pack is not installed"

        with open(path, "rb") as f:
            src_bytes = f.read()

        lines = _outline(src_bytes, lang)

        kinds_filter = args.get("kinds")
        if isinstance(kinds_filter, list) and kinds_filter:
            wanted = {str(k).strip().lower() for k in kinds_filter}
            lines = [ln for ln in lines if ln.strip().split(" ", 1)[0].lower() in wanted]

        if not lines:
            return f"({os.path.basename(path)}: no matching definitions found)"

        header = f"{os.path.basename(path)} [{lang}]"
        body = "\n".join(lines)
        out = f"{header}\n{body}"
        if len(out.encode("utf8")) > _MAX_BYTES:
            out = out.encode("utf8")[:_MAX_BYTES].decode("utf8", "ignore")
            out += "\n... (truncated)"
        return out
    except Exception as e:  # never raise out of a tool
        return f"error: {type(e).__name__}: {e}"


TOOLS = {
    "tree_sitter": {
        "schema": {
            "name": "tree_sitter",
            "description": (
                "Extract a compact nested outline of the definitions in a source "
                "file (functions, classes, methods, structs, enums, interfaces, "
                "types) with their line numbers, using tree-sitter. Use this to "
                "understand a file's structure without reading its full contents. "
                "Supports Python, JS/TS/JSX/TSX, Rust, Go, Java, C/C++, and Ruby."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "file to outline (absolute, or relative to the "
                            "configured project root)"
                        ),
                    },
                    "kinds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "optional filter, e.g. [\"function\",\"class\"]; only "
                            "symbols of these kinds are returned"
                        ),
                    },
                },
                "required": ["path"],
            },
        },
        "run": _run,
    },
}


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    print("=== tree_sitter on voice_input.py (python) ===")
    print(_run({"path": os.path.join(here, "voice_input.py")}))

    # Filtered view: classes only.
    print("\n=== same file, kinds=['class'] ===")
    print(_run({"path": os.path.join(here, "voice_input.py"),
                "kinds": ["class"]}))

    # Try a second language if a sample file exists in the repo.
    for cand in ("voice_input_tools.py", "voice_input_context.py"):
        p = os.path.join(here, cand)
        if os.path.isfile(p):
            print(f"\n=== tree_sitter on {cand} (python) ===")
            print(_run({"path": p}))
            break

    # Unsupported-language path.
    print("\n=== unsupported extension ===")
    print(_run({"path": os.path.join(here, "README.md")}))
