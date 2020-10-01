using DFTK
using Aqua
Aqua.test_all(DFTK, ambiguities=false, stale_deps=(ignore=[:Primes, ], ))
