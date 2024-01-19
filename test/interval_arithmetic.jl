@testitem "Application of an LDA Hamiltonian with Intervals" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using GenericLinearAlgebra
    using IntervalArithmetic: Interval, radius, mid
    silicon = TestCases.silicon

    function discretized_hamiltonian(T, testcase)
        model = model_DFT(convert(Matrix{T}, testcase.lattice), testcase.atoms,
                          testcase.positions, [:lda_x, :lda_c_vwn])

        # For interval arithmetic to give useful numbers,
        # the fft_size should be a power of 2
        Ecut = 10
        fft_size = nextpow.(2, compute_fft_size(model, Ecut))
        basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1), fft_size)

        Hamiltonian(basis; ρ=guess_density(basis))
    end

    T = Float64
    ham    = discretized_hamiltonian(T, silicon)
    hamInt = discretized_hamiltonian(Interval{T}, silicon)
    @test length(ham.basis.model.symmetries) == length(hamInt.basis.model.symmetries)

    hamk = ham.blocks[1]
    hamIntk = hamInt.blocks[1]

    x = randn(Complex{T}, length(G_vectors(ham.basis, ham.basis.kpoints[1])))
    ref = hamk * x
    res = hamIntk * Interval.(x)

    # Small difference between interval arithmetic and normal application
    @test maximum(mid, abs.(res .- ref)) < 1e-9

    # Small error determined by interval arithmetic
    @test maximum(radius, abs.(res)) < 2e-9
end

@testitem "compute_occupation with Intervals" tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using GenericLinearAlgebra
    using IntervalArithmetic: Interval, radius, mid
    testcase = TestCases.silicon

    model = model_LDA(Matrix{Interval{Float64}}(testcase.lattice),
                      testcase.atoms, testcase.positions)
    basis = PlaneWaveBasis(model; Ecut=10, kgrid=(2, 1, 1))

    fermialg = DFTK.default_fermialg(basis.model)
    eigenvalues = [[-0.17268859, 0.26999098, 0.2699912, 0.2699914, 0.35897297, 0.3589743],
                   [-0.08567941, 0.00889772, 0.2246137, 0.2246138, 0.31941655, 0.3870046]]
    evals_inter = Vector{Interval{Float64}}.(eigenvalues)
    occupations, εF = DFTK.compute_occupation(basis, evals_inter, fermialg)

    @test mid(εF) ≈ (eigenvalues[1][4] + eigenvalues[2][5]) / 2 atol=1e-6
    @test mid(sum(DFTK.weighted_ksum(basis, occupations))) ≈ 8.0
end
