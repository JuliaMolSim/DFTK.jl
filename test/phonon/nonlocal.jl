@testsetup module PhononNonlocal
using DFTK

# No exchange-correlation.
function model_tested(lattice::AbstractMatrix, atoms::Vector{<:DFTK.Element},
                      positions::Vector{<:AbstractVector}; kwargs...)
    terms = [Kinetic(),
             AtomicLocal(),
             AtomicNonlocal(),
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
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, PhononNonlocal, TestCases] begin
    using .Phonon: test_frequencies
    using .PhononNonlocal: model_tested

    # Values computed offline with automatic differentiation.
    ω_ref = [ -0.0013321530721379605
               1.221665781102164e-9
               1.6330384820499591e-9
               1.7402445638019258e-9
               0.000576435900440976
               0.0005764359004417748
               0.0005897436031787186
               0.0005897436031787186
               0.0007434725694024692
               0.0008670261586004154
               0.0010523823564599808
               0.0010523823564608157
               0.0010772345814473606
               0.001077234581448384
               0.0015129269010802802
               0.0015129269010835374
               0.0019348246919099461
               0.0019348246919107988 ]

    test_frequencies(model_tested, TestCases.aluminium_primitive; ω_ref)
end

@testitem "Phonon: Local term: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] #=
    =#    setup=[Phonon, PhononNonlocal, TestCases] begin
    using .Phonon: test_frequencies
    using .PhononNonlocal: model_tested
    test_frequencies(model_tested, TestCases.aluminium_primitive)
end
