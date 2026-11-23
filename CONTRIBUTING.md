# Contributing

Please open an issue with the package version, method release ID, benchmark ID,
redacted configuration, exact command and observed error. State whether a model
request was already sent. Never attach API keys, private data or research logs.

For code changes, use a branch or pull request in this public repository. Keep
new fixtures synthetic and preserve method semantics, resource accounting and
failure records. Add a focused regression test before proposing a fix.

```sh
python -m pip install -e '.[dev]'
python -m pytest tests/unit
python -m ruff check .
```

See `.github/workflows/toolkit.yml` for the authoritative CI commands, including
packaging and installed consumer checks. Unit tests and offline checks must not
require secrets or consume model calls. Keep live acceptance opt-in and record
the exact configuration, source identity and request usage separately from
synthetic tests. Follow the upstream licenses for optional code and datasets.
