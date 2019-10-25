using Test
using DFTK

include("testcases.jl")

@testset "High-symmetry kpath construction for silicon" begin
    testcase = silicon
    Ecut = 2
    kline_density = 10

    ref_kcoords = [
        [0.0, 0.0, 0.0], [0.0714286, 0.0, 0.0714286], [0.142857, 0.0, 0.142857],
        [0.214286, 0.0, 0.214286], [0.285714, 0.0, 0.285714], [0.357143, 0.0, 0.357143],
        [0.428571, 0.0, 0.428571], [0.5, 0.0, 0.5], [0.5, 0.0, 0.5],
        [0.5, 0.0625, 0.5625], [0.5, 0.125, 0.625], [0.5, 0.1875, 0.6875],
        [0.5, 0.25, 0.75], [0.5, 0.25, 0.75], [0.458333, 0.291667, 0.75],
        [0.416667, 0.333333, 0.75], [0.375, 0.375, 0.75], [0.375, 0.375, 0.75],
        [0.321429, 0.321429, 0.642857], [0.267857, 0.267857, 0.535714], [0.214286, 0.214286, 0.428571],
        [0.160714, 0.160714, 0.321429], [0.107143, 0.107143, 0.214286], [0.0535714, 0.0535714, 0.107143],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0833333, 0.0833333, 0.0833333],
        [0.166667, 0.166667, 0.166667], [0.25, 0.25, 0.25], [0.333333, 0.333333, 0.333333],
        [0.416667, 0.416667, 0.416667], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
        [0.53125, 0.4375, 0.53125], [0.5625, 0.375, 0.5625], [0.59375, 0.3125, 0.59375],
        [0.625, 0.25, 0.625], [0.625, 0.25, 0.625], [0.583333, 0.25, 0.666667],
        [0.541667, 0.25, 0.708333], [0.5, 0.25, 0.75], [0.5, 0.25, 0.75],
        [0.5, 0.3, 0.7], [0.5, 0.35, 0.65], [0.5, 0.4, 0.6],
        [0.5, 0.45, 0.55], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
        [0.46875, 0.46875, 0.5625], [0.4375, 0.4375, 0.625], [0.40625, 0.40625, 0.6875],
        [0.375, 0.375, 0.75], [0.625, 0.25, 0.625], [0.583333, 0.166667, 0.583333],
        [0.541667, 0.0833333, 0.541667], [0.5, 0.0, 0.5]
    ]

    ref_klabels = Dict(
        "U"=>[0.625, 0.25, 0.625],
        "W"=>[0.5, 0.25, 0.75],
        "X"=>[0.5, 0.0, 0.5],
        "\\Gamma"=>[0.0, 0.0, 0.0],
        "L"=>[0.5, 0.5, 0.5],
        "K"=>[0.375, 0.375, 0.75]
    )

    spec = Species(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_dft(silicon.lattice, :lda_xc_teter93, spec => testcase.positions)
    fft_size = determine_grid_size(model, Ecut)
    basis = PlaneWaveModel(model, fft_size, Ecut, testcase.kcoords, testcase.ksymops)
    kpoints, klabels, kpath = determine_high_symmetry_kpath(basis, kline_density,
                                                            spec => testcase.positions)

    @test length(ref_kcoords) == length(kpoints)
    for ik in 1:length(ref_kcoords)
        @test ref_kcoords[ik] ≈ kpoints[ik].coordinate atol=1e-6
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

    spec = Species(testcase.atnum, psp=load_psp(testcase.psp))
    model = model_dft(silicon.lattice, :lda_xc_teter93, spec => testcase.positions)
    fft_size = determine_grid_size(model, Ecut)
    basis = PlaneWaveModel(model, fft_size, Ecut, testcase.kcoords, testcase.ksymops)

    # Build Hamiltonian just from SAD guess
    ρ0 = guess_gaussian_sad(basis, spec => testcase.positions)
    ham = Hamiltonian(basis, ρ0)


    # Check that plain diagonalisation and compute_bands agree
    diag = diag_lobpcg_hyper()
    prec = PreconditionerKinetic(ham, α=0.1)
    eigres = diag(ham, n_bands + 3, n_conv_check=n_bands, prec=prec, tol=1e-5)

    band_data = compute_bands(ham, basis.kpoints, n_bands)
    for ik in 1:length(basis.kpoints)
        @test eigres.λ[ik][1:n_bands] ≈ band_data.λ[ik] atol=1e-5
    end
end
