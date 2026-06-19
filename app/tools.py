import os
import ast
import operator

DEMO_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_data")

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression. Supports addition, subtraction, multiplication, division, and parentheses. Example: '(42 * 17) + 3'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file from the data directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to read (e.g. 'program_info.txt')",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all available text files in the data directory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

def tool_names():
    """Return the list of tool names defined in TOOL_DEFINITIONS, in order.
    Single source of truth for which per-tool toggles exist."""
    return [t["function"]["name"] for t in TOOL_DEFINITIONS]


# Exponentiation (ast.Pow) and modulo (ast.Mod) are deliberately excluded:
# neither is advertised in the calculator's description (only + - * / and
# parentheses), and Pow is a denial-of-service vector — a model-emitted
# expression like `9**9**9` would spin a CPU and exhaust memory building a
# gigantic integer. Any operator not listed here falls through to the
# "Unsupported expression element" error in _safe_eval, which calculator()
# catches and returns as a harmless "Error: ..." string.
ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return ALLOWED_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPS:
        return ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Largest file read_file will return, in bytes. demo_data/ is curated, so this
# is just a guard against a pathological file being slurped into memory.
MAX_READ_BYTES = 1024 * 1024


def read_file(filename: str) -> str:
    # basename strips any directory components (../, absolute paths) — the first
    # line of defense against traversal.
    safe_name = os.path.basename(filename)

    # Only serve .txt files, matching what list_files advertises.
    if not safe_name.lower().endswith(".txt"):
        return f"Error: File '{safe_name}' not found in data directory."

    filepath = os.path.join(DEMO_DATA_DIR, safe_name)

    # Defense-in-depth: resolve symlinks and confirm the real path is still
    # inside DEMO_DATA_DIR. Guards against a symlink planted in demo_data/.
    base = os.path.realpath(DEMO_DATA_DIR)
    real = os.path.realpath(filepath)
    try:
        contained = os.path.commonpath([real, base]) == base
    except ValueError:
        # Different drives (Windows) -> not contained.
        contained = False
    if not contained or not os.path.isfile(real):
        return f"Error: File '{safe_name}' not found in data directory."

    with open(real, "r", encoding="utf-8") as f:
        return f.read(MAX_READ_BYTES)


def list_files() -> str:
    files = [f for f in os.listdir(DEMO_DATA_DIR) if f.endswith(".txt")]
    if not files:
        return "No text files found in data directory."
    return "\n".join(sorted(files))


TOOL_EXECUTORS = {
    "calculator": lambda args: calculator(args.get("expression", "")),
    "read_file": lambda args: read_file(args.get("filename", "")),
    "list_files": lambda args: list_files(),
}


def execute_tool(name: str, arguments: dict) -> str:
    executor = TOOL_EXECUTORS.get(name)
    if not executor:
        return f"Error: Unknown tool '{name}'"
    return executor(arguments)
