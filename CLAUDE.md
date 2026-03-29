# Quickstart for environment setup
Assuming you have [UV](https://docs.astral.sh/uv/) installed:
```sh
uv sync
```

# Run tests
```sh
uv run pytest
```

# Local Documentation Build
```sh
uv run sphinx-build docs docs/_build/html
open docs/_build/html/index.html
```
