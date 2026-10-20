"""Dependency-free test runner (pytest is also supported): python -m tests.run_tests"""
import importlib, pkgutil, sys, time, traceback
import tests


def main() -> int:
    passed = failed = 0
    for m in pkgutil.iter_modules(tests.__path__):
        if not m.name.startswith("test_"):
            continue
        mod = importlib.import_module(f"tests.{m.name}")
        for name in sorted(dir(mod)):
            if name.startswith("test_") and callable(getattr(mod, name)):
                t0 = time.time()
                try:
                    getattr(mod, name)()
                    passed += 1
                    print(f"PASS {m.name}::{name} ({time.time()-t0:.2f}s)")
                except Exception:
                    failed += 1
                    print(f"FAIL {m.name}::{name}")
                    traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
