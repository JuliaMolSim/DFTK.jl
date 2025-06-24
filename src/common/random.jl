"""
Default random number generator for DFTK.
Deterministic across Julia runs and repeated calls unlike `Random.default_rng()`.
"""
function default_rng()
    Xoshiro(42)
end