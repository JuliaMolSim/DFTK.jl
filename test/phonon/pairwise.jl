@testitem "Phonon: Pairwise: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon] begin
    using DFTK

    a = 5.131570667152971
    lattice = a .* [0 1 1; 1 0 1; 1 1 0]
    atoms     = [ElementCoulomb(:Li), ElementCoulomb(:H)]
    positions = [ones(3)/8, -ones(3)/8]
    testcase = (; lattice, atoms, positions)

    symbols   = [:Li, :H]
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((:Li, :H ) => (; ε=1, σ=2),
                  ( :H, :H ) => (; ε=1, σ=2),
                  (:Li, :Li) => (; ε=1, σ=2))
    terms = [PairwisePotential(V, params; max_radius=10)]

    ω_ref = [ -0.17524941818128295
              -0.17524941818128292
              -0.16832870864925772
              -0.1683287086492577
              -0.15195539471448133
              -0.15193439163371011
              -0.15024477964426225
              -0.1474510503442476
              -0.1474510503442476
              -0.1474293810073767
              -0.1474293810073767
              -0.14475429352434355
              -0.1447542935243435
              -0.13232858616165102
              -0.13230326232686676
              -0.13230326232686657
              -0.09377914380407501
              -0.09377914380407495
              -0.05435449538374675
              -0.05435449538374669
              -0.003915427003557917
              -0.003915427003557904
              -0.0033812394777427185
              -1.7065097811479572e-17
              -3.9611338270885374e-18
               1.7013223995880296e-17
               0.013331409088687666
               0.013331409088687688
               0.013352636670909857
               0.013352636670909858
               0.017234067155574892
               0.017254437310419694
               0.030332222284351517
               0.03033222228435154
               0.03334700396064381
               0.03334700396064386 ]

    tol = 1e-4  # low because of the small radius that we use to speed-up computations
    Phonon.test_frequencies(testcase, terms, ω_ref; tol)
end

@testitem "Phonon: Pairwise: comparison to automatic differentiation" #=
    =#    tags=[:phonon, :slow, :dont_test_mpi] setup=[Phonon] begin
    using DFTK

    a = 5.131570667152971
    lattice = a .* [0 1 1; 1 0 1; 1 1 0]
    atoms     = [ElementCoulomb(:Li), ElementCoulomb(:H)]
    positions = [ones(3)/8, -ones(3)/8]
    testcase = (; lattice, atoms, positions)

    symbols   = [:Li, :H]
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((:Li, :H ) => (; ε=1, σ=2),
                  ( :H, :H ) => (; ε=1, σ=2),
                  (:Li, :Li) => (; ε=1, σ=2))
    terms = [PairwisePotential(V, params; max_radius=10)]

    tol = 1e-4  # low because of the small radius that we use to speed-up computations
    Phonon.test_rand_frequencies(testcase, terms; tol)
end
