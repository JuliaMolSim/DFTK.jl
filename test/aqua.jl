using DFTK
using Aqua
# TODO For now disable type piracy check, as we use that at places to patch up missing functionality.
#      Should disable this on a more fine-grained scale.
Aqua.test_all(DFTK, ambiguities=false, piracy=false, stale_deps=(ignore=[:Primes, ], ))
