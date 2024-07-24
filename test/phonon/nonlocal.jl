@testmodule PhononNonlocal begin
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

@testitem "Phonon: Nonlocal term: comparison to ref testcase" #=
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

@testitem "Phonon: Nonlocal term: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] #=
    =#    setup=[Phonon, PhononNonlocal, TestCases] begin
    using .Phonon: test_frequencies
    using .PhononNonlocal: model_tested
    test_frequencies(model_tested, TestCases.aluminium_primitive)
end

@testitem "Phonon: LDA: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, PhononNonlocal, TestCases] begin
    using .Phonon: test_frequencies

    # Values computed offline with automatic differentiation.
    ω_ref = [ -0.002394568935772381
              -0.0009483244516830787
              -0.0009483244516742963
              -0.0007011496681061224
              -2.3342510049395543e-6
               1.2772524101950583e-6
               9.236368313346967e-6
               0.0004853028802285156
               0.00048530288023371785
               0.0005162064130028024
               0.0005170848409142678
               0.0006579053502793632
               0.0008427616671355543
               0.0008427616671396212
               0.0012763347783044433
               0.0012763347783092767
               0.0015604654945598588
               0.0015609820766927637 ]

    test_frequencies(model_LDA, TestCases.aluminium_primitive; ω_ref)
end

@testitem "Phonon: LDA: NLCC not implemented" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, PhononNonlocal, TestCases] begin
    using .Phonon: test_frequencies
    aluminium_primitive = TestCases.aluminium_primitive
    aluminium_primitive = merge(aluminium_primitive,
                                (; atoms=fill(ElementPsp(aluminium_primitive.atnum,
                                                         psp=load_psp(aluminium_primitive.psp_upf)), 1)))
    @test_throws ErrorException test_frequencies(model_LDA, aluminium_primitive)
end
