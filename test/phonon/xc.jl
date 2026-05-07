@testitem "Phonon: LDA: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, TestCases] begin
    using .Phonon: test_frequencies
    using DFTK

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

    model_LDA(args...; kwargs...) = model_DFT(args...; functionals=LDA(), kwargs...)
    test_frequencies(model_LDA, TestCases.aluminium_primitive; ω_ref)
end

@testitem "Phonon: LDA+NLCC: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, TestCases] begin
    using .Phonon: test_frequencies
    using DFTK

    # Values computed offline with automatic differentiation.
    ω_ref = [ -0.002291246044167315
              -0.0008322141414252373
              -0.0008322141414235875
              -0.0006277141258580046
              -9.265838548786679e-9
               2.7728753822878845e-9
               1.723854614577606e-8
               0.0005132198029985638
               0.000513219803002495
               0.0005311935317559987
               0.0005311935317559987
               0.0006681073906670919
               0.0008491725632174406
               0.0008491725632198118
               0.0012978110205401107
               0.001297811020540865
               0.0015922296328008808
               0.0015922296328024783  ]

    Al = ElementPsp(:Al, load_psp(TestCases.aluminium_primitive.psp_upf))
    aluminium_primitive = merge(TestCases.aluminium_primitive, (; atoms=[Al]))
    model_LDA(args...; kwargs...) = model_DFT(args...; functionals=LDA(), kwargs...)
    test_frequencies(model_LDA, aluminium_primitive; ω_ref)
end

@testitem "Phonon: LDA+NLCC: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, TestCases] begin
    using .Phonon: test_frequencies
    using DFTK
    Al = ElementPsp(:Al, load_psp(TestCases.aluminium_primitive.psp_upf))
    aluminium_primitive = merge(TestCases.aluminium_primitive, (; atoms=[Al]))
    model_LDA(args...; kwargs...) = model_DFT(args...; functionals=LDA(), kwargs...)
    test_frequencies(model_LDA, aluminium_primitive)
end
