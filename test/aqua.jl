@testitem "Aqua" tags=[:dont_test_mpi] begin
    # TODO For now disable type piracy check, as we use that at places to patch
    #      up missing functionality. Should disable this on a more fine-grained scale.

    using DFTK
    using Aqua
    Aqua.test_all(DFTK;
                  ambiguities=false,
                  piracies=false,
                  deps_compat=(check_extras=false, ),
                  stale_deps=(ignore=[:Primes, ], ))
end
