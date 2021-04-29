using Test
using DFTK

include("testcases.jl")

if mpi_nprocs() == 1  # not easy to distribute
@testset "High-symmetry kpath construction for silicon" begin
    testcase = silicon
    Ecut = 2

    ref_kcoords = [
        [0.000000000000, 0.000000000000, 0.000000000000],
        [0.041666666667, 0.000000000000, 0.041666666667],
        [0.083333333333, 0.000000000000, 0.083333333333],
        [0.125000000000, 0.000000000000, 0.125000000000],
        [0.166666666667, 0.000000000000, 0.166666666667],
        [0.208333333333, 0.000000000000, 0.208333333333],
        [0.250000000000, 0.000000000000, 0.250000000000],
        [0.291666666667, 0.000000000000, 0.291666666667],
        [0.333333333333, 0.000000000000, 0.333333333333],
        [0.375000000000, 0.000000000000, 0.375000000000],
        [0.416666666667, 0.000000000000, 0.416666666667],
        [0.458333333333, 0.000000000000, 0.458333333333],
        [0.500000000000, 0.000000000000, 0.500000000000],
        [0.500000000000, 0.000000000000, 0.500000000000],
        [0.500000000000, 0.041666666667, 0.541666666667],
        [0.500000000000, 0.083333333333, 0.583333333333],
        [0.500000000000, 0.125000000000, 0.625000000000],
        [0.500000000000, 0.166666666667, 0.666666666667],
        [0.500000000000, 0.208333333333, 0.708333333333],
        [0.500000000000, 0.250000000000, 0.750000000000],
        [0.500000000000, 0.250000000000, 0.750000000000],
        [0.475000000000, 0.275000000000, 0.750000000000],
        [0.450000000000, 0.300000000000, 0.750000000000],
        [0.425000000000, 0.325000000000, 0.750000000000],
        [0.400000000000, 0.350000000000, 0.750000000000],
        [0.375000000000, 0.375000000000, 0.750000000000],
        [0.375000000000, 0.375000000000, 0.750000000000],
        [0.346153846154, 0.346153846154, 0.692307692308],
        [0.317307692308, 0.317307692308, 0.634615384615],
        [0.288461538462, 0.288461538462, 0.576923076923],
        [0.259615384615, 0.259615384615, 0.519230769231],
        [0.230769230769, 0.230769230769, 0.461538461538],
        [0.201923076923, 0.201923076923, 0.403846153846],
        [0.173076923077, 0.173076923077, 0.346153846154],
        [0.144230769231, 0.144230769231, 0.288461538462],
        [0.115384615385, 0.115384615385, 0.230769230769],
        [0.086538461538, 0.086538461538, 0.173076923077],
        [0.057692307692, 0.057692307692, 0.115384615385],
        [0.028846153846, 0.028846153846, 0.057692307692],
        [0.000000000000, 0.000000000000, 0.000000000000],
        [0.000000000000, 0.000000000000, 0.000000000000],
        [0.045454545455, 0.045454545455, 0.045454545455],
        [0.090909090909, 0.090909090909, 0.090909090909],
        [0.136363636364, 0.136363636364, 0.136363636364],
        [0.181818181818, 0.181818181818, 0.181818181818],
        [0.227272727273, 0.227272727273, 0.227272727273],
        [0.272727272727, 0.272727272727, 0.272727272727],
        [0.318181818182, 0.318181818182, 0.318181818182],
        [0.363636363636, 0.363636363636, 0.363636363636],
        [0.409090909091, 0.409090909091, 0.409090909091],
        [0.454545454545, 0.454545454545, 0.454545454545],
        [0.500000000000, 0.500000000000, 0.500000000000],
        [0.500000000000, 0.500000000000, 0.500000000000],
        [0.515625000000, 0.468750000000, 0.515625000000],
        [0.531250000000, 0.437500000000, 0.531250000000],
        [0.546875000000, 0.406250000000, 0.546875000000],
        [0.562500000000, 0.375000000000, 0.562500000000],
        [0.578125000000, 0.343750000000, 0.578125000000],
        [0.593750000000, 0.312500000000, 0.593750000000],
        [0.609375000000, 0.281250000000, 0.609375000000],
        [0.625000000000, 0.250000000000, 0.625000000000],
        [0.625000000000, 0.250000000000, 0.625000000000],
        [0.600000000000, 0.250000000000, 0.650000000000],
        [0.575000000000, 0.250000000000, 0.675000000000],
        [0.550000000000, 0.250000000000, 0.700000000000],
        [0.525000000000, 0.250000000000, 0.725000000000],
        [0.500000000000, 0.250000000000, 0.750000000000],
        [0.500000000000, 0.250000000000, 0.750000000000],
        [0.500000000000, 0.277777777778, 0.722222222222],
        [0.500000000000, 0.305555555556, 0.694444444444],
        [0.500000000000, 0.333333333333, 0.666666666667],
        [0.500000000000, 0.361111111111, 0.638888888889],
        [0.500000000000, 0.388888888889, 0.611111111111],
        [0.500000000000, 0.416666666667, 0.583333333333],
        [0.500000000000, 0.444444444444, 0.555555555556],
        [0.500000000000, 0.472222222222, 0.527777777778],
        [0.500000000000, 0.500000000000, 0.500000000000],
        [0.500000000000, 0.500000000000, 0.500000000000],
        [0.484375000000, 0.484375000000, 0.531250000000],
        [0.468750000000, 0.468750000000, 0.562500000000],
        [0.453125000000, 0.453125000000, 0.593750000000],
        [0.437500000000, 0.437500000000, 0.625000000000],
        [0.421875000000, 0.421875000000, 0.656250000000],
        [0.406250000000, 0.406250000000, 0.687500000000],
        [0.390625000000, 0.390625000000, 0.718750000000],
        [0.375000000000, 0.375000000000, 0.750000000000],
        [0.625000000000, 0.250000000000, 0.625000000000],
        [0.600000000000, 0.200000000000, 0.600000000000],
        [0.575000000000, 0.150000000000, 0.575000000000],
        [0.550000000000, 0.100000000000, 0.550000000000],
        [0.525000000000, 0.050000000000, 0.525000000000],
        [0.500000000000, 0.000000000000, 0.500000000000],
    ]

    ref_klabels = Dict(
        "U"=>[0.625, 0.25, 0.625],
        "W"=>[0.5, 0.25, 0.75],
        "X"=>[0.5, 0.0, 0.5],
        "Γ"=>[0.0, 0.0, 0.0],
        "L"=>[0.5, 0.5, 0.5],
        "K"=>[0.375, 0.375, 0.75]
    )

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_DFT(silicon.lattice, [spec => testcase.positions], [:lda_xc_teter93])
    kcoords, klabels, kpath = high_symmetry_kpath(model; kline_density=10)

    @test length(ref_kcoords) == length(kcoords)
    for ik in 1:length(ref_kcoords)
        @test ref_kcoords[ik] ≈ kcoords[ik] atol=1e-11
    end

    @test length(klabels) == length(ref_klabels)
    for key in keys(ref_klabels)
        @test klabels[key] ≈ ref_klabels[key] atol=1e-15
    end

    @test kpath[1] == [raw"\Gamma", "X", "W", "K", raw"\Gamma", "L", "U", "W", "L", "K"]
    @test kpath[2] == ["U", "X"]
end

@testset "Compute bands for silicon" begin
    testcase = silicon
    Ecut = 7
    n_bands = 8

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_DFT(silicon.lattice, [spec => testcase.positions], :lda_xc_teter93)
    basis = PlaneWaveBasis(model, Ecut, testcase.kcoords, testcase.ksymops)

    # Build Hamiltonian just from SAD guess
    ρ0 = guess_density(basis, [spec => testcase.positions])
    ham = Hamiltonian(basis; ρ=ρ0)

    # Check that plain diagonalization and compute_bands agree
    eigres = diagonalize_all_kblocks(lobpcg_hyper, ham, n_bands + 3, n_conv_check=n_bands,
                                     tol=1e-5)

    band_data = compute_bands(basis, ρ0, [k.coordinate for k in basis.kpoints], n_bands)
    for ik in 1:length(basis.kpoints)
        @test eigres.λ[ik][1:n_bands] ≈ band_data.λ[ik] atol=1e-5
    end
end

@testset "prepare_band_data" begin
    testcase = silicon
    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_DFT(silicon.lattice, [spec => testcase.positions], :lda_xc_teter93)

    # k coordinates simulating two band branches, Γ => X => W and U => X
    kcoords = [
        [0.000, 0.000, 0.000],
        [0.250, 0.000, 0.250],
        [0.500, 0.000, 0.500],
        #
        [0.500, 0.000, 0.500],
        [0.500, 0.125, 0.625],
        [0.500, 0.250, 0.750],
        #
        [0.625, 0.250, 0.625],
        [0.575, 0.150, 0.575],
        [0.500, 0.000, 0.500],
    ]
    ksymops = [[DFTK.identity_symop()] for _ in 1:length(kcoords)]
    basis = PlaneWaveBasis(model, 5, kcoords, ksymops)
    klabels = Dict("Γ" => [0, 0, 0], "X" => [0.5, 0.0, 0.5],
                   "W" => [0.5, 0.25, 0.75], "U" => [0.625, 0.25, 0.625])

    # Setup some dummy data
    λ = [10ik .+ collect(1:4) for ik = 1:length(kcoords)]  # Simulate 4 computed bands
    λerror = [λ[ik]./100 for ik = 1:length(kcoords)]       # ... and 4 errors

    ret = DFTK.prepare_band_data((basis=basis, λ=λ, λerror=λerror), klabels=klabels)

    @test ret.n_spin   == 1
    @test ret.n_kcoord == 9
    @test ret.n_bands  == 4

    @test ret.branches[1].kindices == [1, 2, 3]
    @test ret.branches[2].kindices == [4, 5, 6]
    @test ret.branches[3].kindices == [7, 8, 9]

    @test ret.branches[1].klabels == ("Γ", "X")
    @test ret.branches[2].klabels == ("X", "W")
    @test ret.branches[3].klabels == ("U", "X")

    for iband in 1:4
        @test ret.branches[1].λ[:, iband, 1] == [10ik .+ iband for ik in 1:3]
        @test ret.branches[2].λ[:, iband, 1] == [10ik .+ iband for ik in 4:6]
        @test ret.branches[3].λ[:, iband, 1] == [10ik .+ iband for ik in 7:9]

        for ibr in 1:3
            @test ret.branches[ibr].λerror[:, iband, 1] == ret.branches[ibr].λ[:, iband, 1] ./ 100
        end
    end

    B = model.recip_lattice
    ref_kdist = zeros(3, 3)  # row idx is kpoint, col idx is branch,
    ikpt = 1
    for ibr in 1:3
        ibr != 1 && (ref_kdist[1, ibr] = ref_kdist[end, ibr-1])
        ikpt += 1
        for ik in 2:3
            ref_kdist[ik, ibr] = (
                ref_kdist[ik-1, ibr] + norm(B * (kcoords[ikpt-1] - kcoords[ikpt]))
            )
            ikpt += 1
        end
    end
    for ibr in 1:3
        @test ret.branches[ibr].kdistances == ref_kdist[:, ibr]
    end

    @test ret.ticks.labels == ["Γ", "X", "W | U", "X"]
    @test ret.ticks.distances == [0.0, ref_kdist[end, 1], ref_kdist[end, 2], ref_kdist[end, 3]]
end

@testset "is_metal" begin
    testcase = silicon
    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_LDA(silicon.lattice, [spec => testcase.positions])

    basis = PlaneWaveBasis(model, 5, testcase.kcoords, testcase.ksymops)
    λ = [[1, 2, 3, 4], [1, 1.5, 3.5, 4.2], [1, 1.1, 3.2, 4.3], [1, 2, 3.3, 4.1]]

    @test !DFTK.is_metal((λ=λ, basis=basis), 2.5)
    @test DFTK.is_metal((λ=λ, basis=basis), 3.2)
end
end
