# Torch-Wrapper

A wrapper for calling `scipy.optimize.minimize` on torch Module

[wiki/test-functions-for-optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization)

install

```bash
pip install git+https://github.com/husisy/torch-wrapper.git

# or you may git clone the repo and install it locally
git clone git+https://github.com/husisy/torch-wrapper.git
pip install .

# for developer
pip install -e .
```

quickstart

```bash
cd example
python draft00.py
# Rastrigin function
# Ackley function
# Rosenbrock function
# Beale function
```

unittest

```bash
pytest
```
