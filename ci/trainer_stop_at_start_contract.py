"""Regression contract for the real Trainer control-flow source.

A standard-library-only structural contract. Integration tests exercise the
runtime path separately; this does not pretend to emulate a Trainer.
"""
import ast
from pathlib import Path

p = Path("src/transformers/trainer.py")
tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
fn = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "_inner_training_loop")

def is_call(node, attr):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    )

begin_index = next(
    i for i, n in enumerate(fn.body)
    if isinstance(n, ast.Assign) and is_call(n.value, "on_train_begin")
)
after = fn.body[begin_index + 1:]
assert after, "No path after on_train_begin"
early_guard = after[0]
assert isinstance(early_guard, ast.If), "First statement after on_train_begin must honor stop flag"
assert ast.unparse(early_guard.test) == "self.control.should_training_stop"
assert len(early_guard.body) == 1
statement = early_guard.body[0]
assert isinstance(statement, ast.Return) and is_call(statement.value, "_finalize_training")
assert [ast.unparse(a) for a in statement.value.args] == [
    "trial", "num_train_samples", "start_time"
]
assert any(
    isinstance(n, ast.If) and "eval_on_start" in ast.unparse(n.test) for n in after[1:]
)
assert any(isinstance(n, ast.For) for n in after[1:])
print("PASS: Trainer on_train_begin stop guard precedes start-eval, epoch loop and optimizer work")
print("PASS: zero-step exit still executes standard training finalization")
