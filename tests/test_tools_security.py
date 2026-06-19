"""Security/hardening tests for app.tools (findings 1 and 3).

Like the other suites these import app.tools directly and never load a model.

Run with:  python -m pytest tests/
"""

from app.tools import calculator, read_file


# --- Finding 1: calculator only supports the advertised + - * / operators -----

def test_basic_arithmetic_still_works():
    assert calculator("2 + 3 * 4") == "14"
    assert calculator("(42 * 17) + 3") == "717"
    assert calculator("-5 + 2") == "-3"
    assert calculator("10 / 4") == "2.5"


def test_exponentiation_is_rejected_not_evaluated():
    """`9**9**9` used to hang building a gigantic integer. Pow is no longer in
    ALLOWED_OPS, so it returns a harmless error string instead (and fast)."""
    result = calculator("9**9**9")
    assert result.startswith("Error:")


def test_modulo_is_rejected():
    """Modulo is undocumented and removed alongside Pow."""
    assert calculator("5 % 2").startswith("Error:")


def test_names_and_calls_are_rejected():
    """No identifiers / function calls survive the AST allowlist."""
    assert calculator("__import__('os')").startswith("Error:")
    assert calculator("pow(9, 9)").startswith("Error:")


# --- Finding 3: read_file containment and extension allowlist ------------------

def test_reads_a_real_demo_file():
    content = read_file("program_info.txt")
    assert not content.startswith("Error:")
    assert len(content) > 0


def test_traversal_is_rejected():
    """basename strips the path; the result is not a real .txt in demo_data."""
    assert read_file("../../etc/passwd").startswith("Error:")
    assert read_file("../tools.py").startswith("Error:")


def test_absolute_path_is_rejected():
    assert read_file("/etc/passwd").startswith("Error:")
    assert read_file(r"C:\Windows\win.ini").startswith("Error:")


def test_non_txt_extension_is_rejected():
    """Only .txt files are served, matching what list_files advertises."""
    assert read_file("program_info.csv").startswith("Error:")
    assert read_file("secrets").startswith("Error:")
