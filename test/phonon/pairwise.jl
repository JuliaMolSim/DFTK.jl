@testmodule PhononPairwise begin
using DFTK

function model_tested(lattice::AbstractMatrix, atoms::Vector{<:DFTK.Element},
                      positions::Vector{<:AbstractVector}; kwargs...)
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((:Li, :H ) => (; ε=1, σ=2),
                  ( :H, :H ) => (; ε=1, σ=2),
                  (:Li, :Li) => (; ε=1, σ=2))
    terms = [Kinetic(),
             PairwisePotential(V, params; max_radius=10)]
    if :temperature in keys(kwargs) && kwargs[:temperature] != 0
        terms = [terms..., Entropy()]
    end
    Model(lattice, atoms, positions; model_name="atomic", terms, n_electrons=length(atoms),
          disable_electrostatics_check=true, kwargs...)
end

testcase = let
    a = 5.131570667152971
    lattice = a .* [0 1 1; 1 0 1; 1 1 0]
    atoms     = [ElementCoulomb(:Li), ElementCoulomb(:H)]
    positions = [ones(3)/8, -ones(3)/8]
    (; lattice, atoms, positions, temperature=0.1)
end
end

@testitem "Phonon: Pairwise: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, PhononPairwise] begin
    using DFTK
    using .Phonon: test_frequencies
    using .PhononPairwise: model_tested, testcase

    ω_ref = [ -0.007027791271072399
              -0.007027791271072398
              -0.006900536310814988
              -0.006900536310814986
              -0.00688581863804306
              -0.006767457623283974
              -0.006767457623283973
              -0.006671943633991508
              -0.0066712169205625236
              -0.006612886568974899
              -0.006612886568974898
              -0.0066121215689177466
              -0.006612121568917746
              -0.006421710386603084
              -0.006421095893096361
              -0.006421095893096357
              -0.006380963485451198
              -0.006380963485451197
              -0.0023188854133402996
              -0.002318885413340299
              -0.0007296417626126479
              -0.0007296417626126462
              -0.0006789096691398658
              -5.658889649128784e-11
               3.0176579159771366e-11
               5.8144997735465574e-11
               0.0013905952690850048
               0.001390595269085006
               0.001391760812737551
               0.0013917608127375512
               0.0015908508495691618
               0.0015918541742743959
               0.002147722981302167
               0.0021477229813021688
               0.0022578909252519617
               0.002257890925251963 ]

    tol = 1e-4  # low because of the small radius that we use to speed-up computations
    test_frequencies(model_tested, testcase; ω_ref, tol)
end

@testitem "Phonon: Pairwise: comparison to automatic differentiation" #=
    =#    tags=[:phonon, :slow, :dont_test_mpi] setup=[Phonon, PhononPairwise] begin
    using DFTK
    using .Phonon: test_frequencies
    using .PhononPairwise: model_tested, testcase

    tol = 1e-4  # low because of the small radius that we use to speed-up computations
    test_frequencies(model_tested, testcase; tol, randomize=true)
end
