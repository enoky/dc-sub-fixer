"""Guard the rule that cross-thread Qt signals must land on bound slots.

Qt decides between a direct and a queued connection from the receiver's thread
affinity, and it can only determine that for a slot belonging to a QObject. A
lambda or a closure is not one, so it gets a direct connection and runs in the
*emitting* thread. When the emitter is a worker and the slot touches widgets,
that is a write to the GUI from the wrong thread: an access violation that
takes the process down with no Python traceback.

That happened twice here, and neither instance was visible on reading the code
— both looked like ordinary signal wiring. The check is done on the source
rather than at runtime so it needs neither Qt nor a display.

Connections to signals of same-thread objects (buttons, sliders) are fine with
a lambda and are deliberately not covered.
"""

import ast
import os

import pytest

GUI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "dcsubfixer", "gui.py")

# Objects whose signals are emitted from a worker thread.
WORKER_HINTS = ("worker", "_render_worker", "_proc_worker")


def _connect_calls(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"):
            yield node


def _receiver_source(node, source):
    """The expression the signal belongs to, e.g. 'worker.finished'."""
    return ast.get_source_segment(source, node.func.value) or ""


@pytest.fixture(scope="module")
def parsed():
    with open(GUI, "r", encoding="utf-8") as fh:
        source = fh.read()
    return source, ast.parse(source)


def test_worker_signals_never_connect_to_a_lambda(parsed):
    source, tree = parsed
    offenders = []
    for call in _connect_calls(tree):
        receiver = _receiver_source(call, source)
        if not any(h in receiver for h in WORKER_HINTS):
            continue
        for arg in call.args[:1]:
            if isinstance(arg, ast.Lambda):
                offenders.append(f"line {call.lineno}: {receiver}.connect(lambda …)")
    assert not offenders, (
        "cross-thread signal connected to a lambda, which Qt will run in the "
        "worker thread:\n  " + "\n  ".join(offenders)
    )


def test_worker_signals_connect_to_attributes_of_self(parsed):
    """Bound methods of the window, so Qt can see they live on the GUI thread."""
    source, tree = parsed
    checked = 0
    for call in _connect_calls(tree):
        receiver = _receiver_source(call, source)
        if not any(h in receiver for h in WORKER_HINTS):
            continue
        if not call.args:
            continue
        arg = call.args[0]
        target = ast.get_source_segment(source, arg) or ""
        # worker.run is the one deliberate exception: it belongs to the worker
        # and is meant to execute in the worker thread.
        if target.endswith(".run"):
            continue
        assert isinstance(arg, ast.Attribute), (
            f"line {call.lineno}: {receiver}.connect({target}) should be a bound "
            "method so Qt gives it a queued connection"
        )
        assert target.startswith("self."), (
            f"line {call.lineno}: {receiver}.connect({target}) should be a slot "
            "on the window"
        )
        checked += 1
    assert checked >= 4, f"expected to find the worker connections, saw {checked}"
