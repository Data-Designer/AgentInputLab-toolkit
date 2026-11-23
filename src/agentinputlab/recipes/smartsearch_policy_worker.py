"""Fresh-process, distributed native SmartSearch policy-stage entrypoint."""
import sys

from .smartsearch_policy import run_native_policy_rank

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("expected a create-only policy output root and pinned plan SHA-256")
    run_native_policy_rank(sys.argv[1], sys.argv[2])
