# Interactive Proofs for Quantum Learning

Do not be fooled! This is **not** to the level of complexity of frontier ML. In fact, we are actually learning parity with noise! This is an NP-hard problem typically.

# Quickstart for environment setup
Assuming you have [UV](https://docs.astral.sh/uv/) installed:
```sh
uv sync
```

# Local Documentation
```sh
uv run sphinx-build docs docs/_build/html
open docs/_build/html/index.html
```

# Useful resources

See the following papers:
- [Matthias C. Caro et al. “Classical Verification of Quantum Learning”](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITCS.2024.24)
- [Urmila Mahadev. Classical Verification of Quantum Computations](https://arxiv.org/abs/1804.01082) 
- [Matthias C. Caro et al. Interactive proofs for verifying (quantum) learning and testing](https://arxiv.org/abs/2410.23969)

# TODO:
- Clean up slurm scripts, probably integrate into the harness or otherwise
- Re-run experiments

