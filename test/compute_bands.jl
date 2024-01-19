@testitem "High-symmetry kpath construction for silicon" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using Brillouin: interpolate
    testcase = TestCases.silicon

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
        :U => [0.625, 0.25, 0.625],
        :W => [0.5, 0.25, 0.75],
        :X => [0.5, 0.0, 0.5],
        :Γ => [0.0, 0.0, 0.0],
        :L => [0.5, 0.5, 0.5],
        :K => [0.375, 0.375, 0.75]
    )

    model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions)
    kpath = irrfbz_path(model)

    @test length(kpath.points) == length(ref_klabels)
    for key in keys(ref_klabels)
        @test kpath.points[key] ≈ ref_klabels[key] atol=1e-15
    end
    @test kpath.paths[1] == [:Γ, :X, :U]
    @test kpath.paths[2] == [:K, :Γ, :L, :W, :X]

    # Interpolate the path and check
    kinter = interpolate(kpath, density=22.7)
    @test ref_kcoords ≈ kinter  atol=1e-11
    @test length.(kinter.kpaths) == [18, 42]
end

@testitem "High-symmetry kpath construction for 1D system" tags=[:dont_test_mpi] begin
    using DFTK
    using Brillouin: interpolate
    using LinearAlgebra

    lattice = diagm([8.0, 0, 0])
    model   = Model(lattice; terms=[Kinetic()])
    kpath   = irrfbz_path(model)

    @test length(kpath.paths)  == 1
    @test length(kpath.points) == 2
    @test kpath.paths == [[:Γ, :X]]
    @test kpath.points[:Γ] == [0.0]
    @test kpath.points[:X] == [0.5]

    kinter = interpolate(kpath, density=20)
    @test length(kinter) == 8
    @test kinter[1] == [0.0]
    @test kinter[8] == [0.5]
end

@testitem "Compute bands for silicon" tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using Brillouin: interpolate
    testcase = TestCases.silicon

    Ecut = 7
    n_bands = 8

    model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions)
    kgrid = ExplicitKpoints(interpolate(irrfbz_path(model), density=3))
    basis = PlaneWaveBasis(model; Ecut, kgrid)

    # Check that plain diagonalization and compute_bands agree
    ρ   = guess_density(basis)
    ham = Hamiltonian(basis; ρ)
    band_data = compute_bands(basis, kgrid; ρ, n_bands)

    eigres = diagonalize_all_kblocks(lobpcg_hyper, ham, n_bands + 3,
                                     n_conv_check=n_bands, tol=1e-5)
    for ik = 1:length(basis.kpoints)
        @test eigres.λ[ik][1:n_bands] ≈ band_data.eigenvalues[ik] atol=1e-5
    end
end

@testitem "prepare_band_data" tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using Brillouin: interpolate
    using LinearAlgebra
    testcase = TestCases.silicon

    model    = model_LDA(testcase.lattice, testcase.atoms, testcase.positions)
    kpath    = irrfbz_path(model)
    kinter   = interpolate(irrfbz_path(model), density=3)
    basis    = PlaneWaveBasis(model; Ecut=5, kgrid=ExplicitKpoints(kinter))

    # Setup some dummy data
    eigenvalues = [10ik .+ collect(1:4)        for ik = 1:length(kinter)]
    eigenvalues_error = [eigenvalues[ik]./100  for ik = 1:length(kinter)]
    band_data = (; basis, eigenvalues, eigenvalues_error, kinter)
    ret = DFTK.data_for_plotting(band_data)

    @test ret.n_spin   == 1
    @test ret.n_kcoord == 8
    @test ret.n_bands  == 4

    for iband = 1:4
        @test ret.eigenvalues[:, iband, 1] == [10ik .+ iband for ik = 1:8]
        @test ret.eigenvalues_error[:, iband, 1] == ret.eigenvalues[:, iband, 1] ./ 100
    end

    B = model.recip_lattice
    ref_kdist = [0.0]
    for ik = 2:8
        if ik != 4
            push!(ref_kdist, ref_kdist[end] + norm(B * (kinter[ik-1] - kinter[ik])))
        else
            # At ik = 6, the branch changes so kdistance does not increase.
            push!(ref_kdist, ref_kdist[end])
        end
    end
    @test ret.kdistances ≈ ref_kdist atol=1e-14
    @test ret.ticks.labels == ["Γ", "X", "U | K", "Γ", "L", "W", "X"]
    @test ret.ticks.distances ≈ ref_kdist[[1, 2, 3, 5, 6, 7, 8]] atol=1e-14
    @test ret.kbranches == [1:3, 4:8]
end

@testitem "is_metal" tags=[:dont_test_mpi] begin
    using DFTK

    λ = [[1, 2, 3, 4], [1, 1.5, 3.5, 4.2], [1, 1.1, 3.2, 4.3], [1, 2, 3.3, 4.1]]
    @test !DFTK.is_metal(λ, 2.5)
    @test  DFTK.is_metal(λ, 3.2)
end

@testitem "High-symmetry kpath for nonstandard lattice" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using Brillouin: interpolate
    testcase = TestCases.silicon

    lattice_std = [0 1 1; 1 0 1; 1 1 0] .* 5.13
    model_std   = model_LDA(lattice_std, testcase.atoms, testcase.positions)

    # Non-standard lattice parameters that describe the same system as model_standard.
    lattice_nst = copy(lattice_std)
    lattice_nst[:, 3] .+= lattice_nst[:, 1] .* 3
    position_nst = [[-2, 1, 1]/8, -[-2, 1, 1]/8]
    model_nst = model_LDA(lattice_nst, testcase.atoms, position_nst)

    kpath_std = irrfbz_path(model_std)
    kpath_nst = irrfbz_path(model_nst)
    @test Set(keys(kpath_std.points)) == Set(keys(kpath_nst.points))
    @test kpath_std.paths == kpath_nst.paths

    # Check the k points are the same in Cartesian coordinates.
    kinter_std = interpolate(kpath_std; density=20)
    kinter_nst = interpolate(kpath_nst; density=20)
    for (k_std, k_nst) in zip(kinter_std, kinter_nst)
        @test(  model_std.recip_lattice * k_std
              ≈ model_nst.recip_lattice * k_nst)
    end
end
