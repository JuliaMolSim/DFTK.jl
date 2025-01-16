@testitem "Aqua" tags=[:dont_test_mpi, :dont_test_windows] begin
    # TODO For now disable type piracy check, as we use that at places to patch
    #      up missing functionality. Should disable this on a more fine-grained scale.

    using DFTK
    using Aqua
    using LinearAlgebra
    Aqua.test_all(DFTK;
                  ambiguities=(; exclude=[
                               # Type piracies we do for FFT stuff
                               *, \, mul!, dot, ldiv!,
                               # Libxc stuff
                               DFTK.potential_terms, DFTK.kernel_terms]),
                  piracies=false,
                  deps_compat=(; check_extras=(; ignore=[:CUDA_Runtime_jll])),
                  stale_deps=(; ignore=[:Primes, ]),
                  persistent_tasks=(; broken=VERSION < v"1.11"),
                 )
end
