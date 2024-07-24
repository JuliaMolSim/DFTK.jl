@testmodule PhononLocal begin
using DFTK

# No exchange-correlation and only a local potential.
function model_tested(lattice::AbstractMatrix, atoms::Vector{<:DFTK.Element},
                      positions::Vector{<:AbstractVector}; kwargs...)
    terms = [Kinetic(),
             AtomicLocal(),
             Ewald(),
             PspCorrection(),
             Hartree()]
    if :temperature in keys(kwargs) && kwargs[:temperature] != 0
        terms = [terms..., Entropy()]
    end
    Model(lattice, atoms, positions; model_name="atomic", terms, kwargs...)
end
end

@testitem "Phonon: Local term: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, PhononLocal, TestCases] begin
    using .Phonon: test_frequencies
    using .PhononLocal: model_tested

    # Values computed offline with automatic differentiation.
    ω_ref = [ -3.6569888415715e-9
              -3.6569888415715e-9
              -2.263180017613055e-9
               0.000443073786433812
               0.0004675174987222679
               0.00046751749874345965
               0.000520667604960504
               0.0005206676049755671
               0.0008481450680251938
               0.0009079870302639688
               0.0009079870302721681
               0.0010121409655813906
               0.0010121409655813906
               0.0013408306319911576
               0.0013779547317006979
               0.001377954731723582
               0.0014021878602703752
               0.001402187860292344 ]
    test_frequencies(model_tested, TestCases.aluminium_primitive; ω_ref)
end

@testitem "Phonon: Local term: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, PhononLocal, TestCases] begin
    using .Phonon: test_frequencies
    using .PhononLocal: model_tested
    test_frequencies(model_tested, TestCases.aluminium_primitive)
end
