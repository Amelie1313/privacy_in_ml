# privacy_in_ml

## Setup

This project requires **Python 3.11** (see `.python-version`). Newer Python versions (3.12+, especially 3.14) can resolve to a scikit-learn build that's incompatible with `diffprivlib` and will fail with:

```
ImportError: cannot import name 'DOUBLE' from 'sklearn.tree._tree'
```

To set up a working environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins exact package versions (in particular `scikit-learn==1.7.2` alongside `diffprivlib==0.6.6`) so everyone gets the same, working combination. Don't casually bump `scikit-learn` - re-test that the internal perturbation sections (which use `diffprivlib`) still import and run before changing the pin.
