"""Where the repository keeps things.

Every script imports these rather than deriving paths from its own location, so
the corpus, the results and the figures always land where the README says.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

for _p in (DATA, RESULTS, FIGURES):
    os.makedirs(_p, exist_ok=True)
