using Test
using DFTK

include("testcases.jl")

if mpi_nprocs() == 1  # not easy to distribute
@testset "High-symmetry kpath construction for silicon" begin
    testcase = silicon
    Ecut = 2

    ref_kcoords = [
        [0.000000000000, 0.000000000000, 0.000000000000],
        [0.038461538462, 0.000000000000, 0.038461538462],
        [0.076923076923, 0.000000000000, 0.076923076923],
        [0.115384615385, 0.000000000000, 0.115384615385],
        [0.153846153846, 0.000000000000, 0.153846153846],
        [0.192307692308, 0.000000000000, 0.192307692308],
        [0.230769230769, 0.000000000000, 0.230769230769],
        [0.269230769231, 0.000000000000, 0.269230769231],
        [0.307692307692, 0.000000000000, 0.307692307692],
        [0.346153846154, 0.000000000000, 0.346153846154],
        [0.384615384615, 0.000000000000, 0.384615384615],
        [0.423076923077, 0.000000000000, 0.423076923077],
        [0.461538461538, 0.000000000000, 0.461538461538],
        [0.500000000000, 0.000000000000, 0.500000000000],
        [0.531250000000, 0.062500000000, 0.531250000000],
        [0.562500000000, 0.125000000000, 0.562500000000],
        [0.593750000000, 0.187500000000, 0.593750000000],
        [0.625000000000, 0.250000000000, 0.625000000000],
        [0.375000000000, 0.375000000000, 0.750000000000],
        [0.348214285714, 0.348214285714, 0.696428571429],
        [0.321428571429, 0.321428571429, 0.642857142857],
        [0.294642857143, 0.294642857143, 0.589285714286],
        [0.267857142857, 0.267857142857, 0.535714285714],
        [0.241071428571, 0.241071428571, 0.482142857143],
        [0.214285714286, 0.214285714286, 0.428571428571],
        [0.187500000000, 0.187500000000, 0.375000000000],
        [0.160714285714, 0.160714285714, 0.321428571429],
        [0.133928571429, 0.133928571429, 0.267857142857],
        [0.107142857143, 0.107142857143, 0.214285714286],
        [0.080357142857, 0.080357142857, 0.160714285714],
        [0.053571428571, 0.053571428571, 0.107142857143],
        [0.026785714286, 0.026785714286, 0.053571428571],
        [0.000000000000, 0.000000000000, 0.000000000000],
        [0.041666666667, 0.041666666667, 0.041666666667],
        [0.083333333333, 0.083333333333, 0.083333333333],
        [0.125000000000, 0.125000000000, 0.125000000000],
        [0.166666666667, 0.166666666667, 0.166666666667],
        [0.208333333333, 0.208333333333, 0.208333333333],
        [0.250000000000, 0.250000000000, 0.250000000000],
        [0.291666666667, 0.291666666667, 0.291666666667],
        [0.333333333333, 0.333333333333, 0.333333333333],
        [0.375000000000, 0.375000000000, 0.375000000000],
        [0.416666666667, 0.416666666667, 0.416666666667],
        [0.458333333333, 0.458333333333, 0.458333333333],
        [0.500000000000, 0.500000000000, 0.500000000000],
        [0.500000000000, 0.472222222222, 0.527777777778],
        [0.500000000000, 0.444444444444, 0.555555555556],
        [0.500000000000, 0.416666666667, 0.583333333333],
        [0.500000000000, 0.388888888889, 0.611111111111],
        [0.500000000000, 0.361111111111, 0.638888888889],
        [0.500000000000, 0.333333333333, 0.666666666667],
        [0.500000000000, 0.305555555556, 0.694444444444],
        [0.500000000000, 0.277777777778, 0.722222222222],
        [0.500000000000, 0.250000000000, 0.750000000000],
        [0.500000000000, 0.208333333333, 0.708333333333],
        [0.500000000000, 0.166666666667, 0.666666666667],
        [0.500000000000, 0.125000000000, 0.625000000000],
        [0.500000000000, 0.083333333333, 0.583333333333],
        [0.500000000000, 0.041666666667, 0.541666666667],
        [0.500000000000, 0.000000000000, 0.500000000000],
    ]

    ref_klabels = Dict(
        :U=>[0.625, 0.25, 0.625],
        :W=>[0.5, 0.25, 0.75],
        :X=>[0.5, 0.0, 0.5],
        :Γ=>[0.0, 0.0, 0.0],
        :L=>[0.5, 0.5, 0.5],
        :K=>[0.375, 0.375, 0.75]
    )

    model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions)
    kcoords, klabels, kpath, kbranches = high_symmetry_kpath(model; kline_density=22.7)

    @test length(ref_kcoords) == length(kcoords)
    for ik in 1:length(ref_kcoords)
        @test ref_kcoords[ik] ≈ kcoords[ik] atol=1e-11
    end

    @test length(klabels) == length(ref_klabels)
    for key in keys(ref_klabels)
        @test klabels[key] ≈ ref_klabels[key] atol=1e-15
    end

    @test kpath[1] == ["Γ", "X", "U"]
    @test kpath[2] == ["K", "Γ", "L", "W", "X"]

    @test kbranches == [1:18, 19:60]
end

@testset "High-symmetry kpath construction for 1D system" begin
    lattice = diagm([8.0, 0, 0])
    model = Model(lattice; n_electrons=1, terms=[Kinetic()])
    kcoords, klabels, kpath, kbranches = high_symmetry_kpath(model; kline_density=20)

    @test length(kcoords) == 17
    @test kcoords[1]  ≈ [-1/2, 0, 0]
    @test kcoords[9]  ≈ [   0, 0, 0]
    @test kcoords[17] ≈ [ 1/2, 0, 0]
    @test length(kpath) == 1
    @test kbranches == [1:17]
end

@testset "Compute bands for silicon" begin
    testcase = silicon
    Ecut = 7
    n_bands = 8

    model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions)
    basis = PlaneWaveBasis(model, Ecut, testcase.kcoords, testcase.kweights)

    # Build Hamiltonian just from SAD guess
    ρ0 = guess_density(basis)
    ham = Hamiltonian(basis; ρ=ρ0)

    # Check that plain diagonalization and compute_bands agree
    eigres = diagonalize_all_kblocks(lobpcg_hyper, ham, n_bands + 3, n_conv_check=n_bands,
                                     tol=1e-5)

    band_data = compute_bands(basis, [k.coordinate for k in basis.kpoints]; ρ=ρ0, n_bands)
    for ik in 1:length(basis.kpoints)
        @test eigres.λ[ik][1:n_bands] ≈ band_data.λ[ik] atol=1e-5
    end
end

@testset "prepare_band_data" begin
    testcase = silicon
    model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions)

    # k coordinates simulating two band branches, Γ => X => W and U => X
    kcoords = [
        [0.000, 0.000, 0.000], # Γ
        [0.250, 0.000, 0.250],
        [0.500, 0.000, 0.500], # X
        [0.500, 0.125, 0.625],
        [0.500, 0.250, 0.750], # W
        #
        [0.625, 0.250, 0.625], # U
        [0.575, 0.150, 0.575],
        [0.500, 0.000, 0.500], # X
    ]
    kweights = ones(9) ./ 9
    basis = PlaneWaveBasis(model, 5, kcoords, kweights)
    klabels = Dict("Γ" => [0, 0, 0], "X" => [0.5, 0.0, 0.5],
                   "W" => [0.5, 0.25, 0.75], "U" => [0.625, 0.25, 0.625])
    kbranches = [1:5, 6:8]

    # Setup some dummy data
    λ = [10ik .+ collect(1:4) for ik = 1:length(kcoords)]  # Simulate 4 computed bands
    λerror = [λ[ik]./100 for ik = 1:length(kcoords)]       # ... and 4 errors

    ret = DFTK.prepare_band_data((basis=basis, λ=λ, λerror=λerror), klabels=klabels, kbranches=kbranches)

    @test ret.n_spin   == 1
    @test ret.n_kcoord == 8
    @test ret.n_bands  == 4

    for iband in 1:4
        @test ret.λ[:, iband, 1] == [10ik .+ iband for ik in 1:8]
        @test ret.λerror[:, iband, 1] == ret.λ[:, iband, 1] ./ 100
    end

    B = model.recip_lattice
    ref_kdist = [0.0]
    for ik in 2:8
        if ik != 6
            push!(ref_kdist, ref_kdist[end] + norm(B * (kcoords[ik-1] - kcoords[ik])))
        else
            # At ik = 6, the branch changes so kdistance does not increase.
            push!(ref_kdist, ref_kdist[end])
        end
    end
    @test ret.kdistances == ref_kdist
    @test ret.ticks.labels == ["Γ", "X", "W | U", "X"]
    @test ret.ticks.distances == ref_kdist[[1, 3, 5, 8]]
    @test ret.kbranches == kbranches
end

@testset "is_metal" begin
    testcase = silicon
    model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions)

    basis = PlaneWaveBasis(model, 5, testcase.kcoords, testcase.kweights)
    λ = [[1, 2, 3, 4], [1, 1.5, 3.5, 4.2], [1, 1.1, 3.2, 4.3], [1, 2, 3.3, 4.1]]

    @test !DFTK.is_metal((λ=λ, basis=basis), 2.5)
    @test DFTK.is_metal((λ=λ, basis=basis), 3.2)
end

@testset "High-symmetry kpath for nonstandard lattice" begin
    testcase = silicon
    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    lattice_standard = [0 1 1; 1 0 1; 1 1 0] .* 5.13
    model_standard = model_LDA(lattice_standard, [spec => [ones(3)/8, -ones(3)/8]])

    # Non-standard lattice parameters that describe the same system as model_standard.
    lattice_nonstandard = copy(lattice_standard)
    lattice_nonstandard[:, 3] .+= lattice_nonstandard[:, 1] .* 3
    model_nonstandard = model_LDA(lattice_nonstandard,
                                  [spec => [[-2, 1, 1]/8, -[-2, 1, 1]/8]])

    data_standard = high_symmetry_kpath(model_standard)
    data_nonstandard = high_symmetry_kpath(model_nonstandard)

    # Check the k points are the same in Cartesian coordinates.
    for (k_standard, k_nonstandard) in zip(data_standard.kcoords, data_nonstandard.kcoords)
        @test(  model_standard.recip_lattice    * k_standard
              ≈ model_nonstandard.recip_lattice * k_nonstandard)
    end
    @test Set(keys(data_standard.klabels)) == Set(keys(data_nonstandard.klabels))
    @test data_standard.kpath == data_nonstandard.kpath
    @test data_standard.kbranches == data_nonstandard.kbranches
end
end

