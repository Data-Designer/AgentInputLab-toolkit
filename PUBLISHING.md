# Publication policy

Registry publication is manual and requires owner authorization after acceptance. The `publish.yml` workflow is dispatched explicitly with a version and full commit SHA; it accepts only `main` after successful cross-version CI for that exact commit. PyPI Trusted Publishing is scoped to this repository, workflow and the `pypi` environment. Ordinary pushes and pull requests never upload packages. The npm package remains private; npm publication is deferred.

This toolkit has its own public repository and independent history. Never merge or mirror a private research repository into it, and never change a research repository's visibility to publish toolkit updates. Export only reviewed library changes; keep research data, paper sources, private configuration and credentials outside this repository.

Before publication: verify account and namespace ownership, review dependency licenses, run offline and installed-package tests, and scan both distributions. Runtime changes also require the relevant real-backend acceptance plan; documentation-only releases reuse the unchanged runtime's existing evidence. Record unsupported, incomplete and untested combinations accurately; an unimplemented adapter is not a scientifically inapplicable combination. See [acceptance protocol](docs/toolkit-acceptance.md).
