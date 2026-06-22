@testitem "Reference energy tests of Coulomb kernels" tags=[:exx,:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using DFTK: exx_energy_only, compute_kernel_fourier
    using FastGaussQuadrature
    using LinearAlgebra
    using .TestCases: silicon

    # TODO: This is a bad test, better test properties, see details at the end of file

    Si = ElementPsp(14, load_psp(silicon.psp_upf))
    model  = model_DFT(silicon.lattice, [Si, Si], silicon.positions; functionals=PBE())
    basis  = PlaneWaveBasis(model; Ecut=10, kgrid=(1, 1, 1))
    scfres = self_consistent_field(basis; tol=1e-10, callback=identity)

    n_occ = 4
    kpt  = basis.kpoints[1]
    ψk   = scfres.ψ[1][:, 1:n_occ]
    occk = scfres.occupation[1][1:n_occ]
    ψk_real = similar(ψk, eltype(ψk), basis.fft_size..., n_occ)
    @views for n = 1:n_occ
        ifft!(ψk_real[:, :, :, n], basis, kpt, ψk[:, n])
    end

    k_probe = compute_kernel_fourier(Coulomb(ProbeCharge()), basis)
    @testset "Coulomb with ProbeCharge" begin
        E_probe = exx_energy_only(basis, kpt, k_probe, ψk_real, occk)
        E_ref = -2.3383063575660987
        @test abs(E_ref - E_probe) < 1e-6
    end

    @testset "Coulomb with ReplaceSingularity" begin
        k_neglect = compute_kernel_fourier(Coulomb(ReplaceSingularity(0.0)), basis)
        E_neglect = exx_energy_only(basis, kpt, k_neglect, ψk_real, occk)
        E_ref = -0.7349457693125514
        @test abs(E_ref - E_neglect) < 1e-6
        @test norm(k_neglect[2:end] - k_probe[2:end]) < 1e-6
    end

    @testset "LongRangeCoulomb with ProbeCharge" begin
        k_lr = compute_kernel_fourier(LongRangeCoulomb(0.1, ProbeCharge()), basis)
        E_lr = exx_energy_only(basis, kpt, k_lr, ψk_real, occk)
        E_ref = -0.44269774759135383
        @test abs(E_ref - E_lr) < 1e-6
    end

    @testset "ShortRangeCoulomb" begin
        k_sr = compute_kernel_fourier(ShortRangeCoulomb(0.1), basis)
        E_sr = exx_energy_only(basis, kpt, k_sr, ψk_real, occk)
        E_ref = -5.384688524633953
        @test abs(E_ref - E_sr) < 1e-6
    end

    @testset "ShortRangeCoulomb plus LongRangeCoulomb is coulomb" begin
        k_lr  = compute_kernel_fourier(LongRangeCoulomb(0.1, ProbeCharge()), basis)
        k_sr  = compute_kernel_fourier(ShortRangeCoulomb(0.1), basis)
        k_sum = k_lr + k_sr

        # Note: The G=0 component does not match up, because in short-range Coulomb
        # we can take the G->0 limit exactly, but in long-range we cannot and instead
        # use the ProbeCharge regularisation. However, for finite k-points the ProbeCharge
        # regularisation for LongRangeCoulomb and Coulomb does not agree, thus we need
        # to exclude the G=0 component from the test below.
        @test maximum(abs, (k_sum - k_probe)[2:end]) < 1e-12
    end

    @testset "SphericallyTruncatedCoulomb" begin
        k_strunc = compute_kernel_fourier(SphericallyTruncatedCoulomb(), basis)
        E_strunc = exx_energy_only(basis, kpt, k_strunc, ψk_real, occk)
        E_ref = -2.360166200435632
        @test abs(E_ref - E_strunc) < 1e-6

        # TODO: Test this gives a spherically truncated function.
    end

    @testset "WignerSeitzTruncatedCoulomb" begin
        k_wstrunc = compute_kernel_fourier(WignerSeitzTruncatedCoulomb(), basis)
        E_wstrunc = exx_energy_only(basis, kpt, k_wstrunc, ψk_real, occk)
        E_ref = -2.3456813523805415
        @test abs(E_ref - E_wstrunc) < 1e-6
    end

    @testset "VoxelAveraged" begin
        k_voxavg = compute_kernel_fourier(Coulomb(VoxelAveraged()), basis)
        E_voxavg = exx_energy_only(basis, kpt, k_voxavg, ψk_real, occk)
        E_ref = -2.249032672407079
        @test abs(E_ref - E_voxavg) < 1e-6
    end
end

@testitem "Reference kernel tests of Coulomb kernels (non-cubic lattices)" #=
        =# tags=[:exx,:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using DFTK: compute_kernel_fourier
    using FastGaussQuadrature
    using LinearAlgebra
    using .TestCases: all_testcases

    basis_Pt = let  # hexagonal
        pt = all_testcases.platinum_hcp
        model = model_DFT(pt.lattice, pt.atoms, pt.positions; functionals=PBE())
        PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
    end
    basis_Ga = let  # orthorhombic
        ga = all_testcases.gallium_orthorhombic
        model = model_DFT(ga.lattice, ga.atoms, ga.positions; functionals=PBE())
        PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
    end
    basis_Sb = let  # rhombohedral
        sb = all_testcases.antimony_rhombohedral
        model = model_DFT(sb.lattice, sb.atoms, sb.positions; functionals=PBE())
        PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
    end
    basis_illcond = let  # Needle-like and very ill-conditioned lattice
        illconditioned = Float64[1.0 1.0 0.0
                                 1.0 1.1 0.0
                                 0.0 0.0 6.0]
        silicon = all_testcases.silicon
        model = model_DFT(illconditioned, silicon.atoms, silicon.positions; functionals=PBE())
        PlaneWaveBasis(model; Ecut=50, kgrid=(1, 1, 1))
        # Ecut=50 is ok because the lattice is ill-conditioned
    end

    @testset "WignerSeitzTruncatedCoulomb" begin
        # TODO: These tests are not super useful, they mainly enable refactoring; they
        #       should really be replaced by tests of properties of WignerSeitzTruncatedCoulomb
        #       against SphericallyTruncatedCoulomb, for example.
        k_wstrunc = compute_kernel_fourier(WignerSeitzTruncatedCoulomb(), basis_Pt)
        @test k_wstrunc[1:5] ≈ [289.8199039694483, 39.805749013785956, 5.580324780990978,
                                4.320510620666685, 1.71839014451522]

        k_wstrunc = compute_kernel_fourier(WignerSeitzTruncatedCoulomb(), basis_Ga)
        @test k_wstrunc[1:5] ≈ [133.48675852141807, 12.54391563903425, 1.0201128711380307,
                                1.0201128711380307, 12.54391563903425]

        k_wstrunc = compute_kernel_fourier(WignerSeitzTruncatedCoulomb(), basis_Sb)
        @test k_wstrunc[1:5] ≈ [133.5869934680494, 17.875585010939407, 3.0560733842605785,
                                1.6670099917149919, 1.6670099917149919] atol=1e-6

        k_wstrunc = compute_kernel_fourier(WignerSeitzTruncatedCoulomb(), basis_illcond)
        @test k_wstrunc[1:5] ≈ [0.6835202703428708, 0.10574224010892719, 0.10574224010892719,
                                0.363208874652315, 0.10081120609383883] atol=1e-6
    end
end


@testitem "Consistency tests of Coulomb-like kernels" tags=[:exx,:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using .TestCases: silicon
    using LinearAlgebra
    using QuadGK

    Si = ElementPsp(14, load_psp(silicon.psp_upf))
    model  = model_DFT(silicon.lattice, [Si, Si], silicon.positions; functionals=PBE())
    basis  = PlaneWaveBasis(model; Ecut=10, kgrid=(1, 1, 1))
    scfres = self_consistent_field(basis; tol=1e-10, callback=identity)

    n_occ = 4
    kpt  = basis.kpoints[1]
    ψk   = scfres.ψ[1][:, 1:n_occ]
    occk = scfres.occupation[1][1:n_occ]
    ψk_real = similar(ψk, eltype(ψk), basis.fft_size..., n_occ)
    @views for n = 1:n_occ
        ifft!(ψk_real[:, :, :, n], basis, kpt, ψk[:, n])
    end

    @testset "Probe-charge integrals" begin
        α = π^2 / basis.Ecut  # width of probe charge

        for kernel in (Coulomb(), LongRangeCoulomb())
            # Compute Brillouin integral of k(q) times probe charge in Fourier (e^(-α q^2))
            # We use that the integrand is spherically symmetric and thus directly introduce
            # a 4π q^2 factor
            qmax = sqrt(316 / α)  # At that point e^(-α q^2) is numerically zero
            numerical_integral, _ = quadgk(0, qmax) do q
                4π * q^2 * DFTK.eval_kernel_fourier(kernel, q^2) * exp(-α * q^2)
            end
            analytical_integral = DFTK.eval_probe_charge_integral(kernel, α)
            @test abs(numerical_integral - analytical_integral) < 1e-12
        end
    end
end

# TODO: Tests ot include in the future are
#       - Test for type stability (i.e. when Float32 is used)
#       - Test properties, such as the fact that
#         the Coulomb energy converges to the same value as the supercell / k-grid
#         more suprecells are used


@testitem "Asymptotic consistency of interaction kernels for localized density" tags=[:exx,:dont_test_mpi] begin
    using DFTK
    using DFTK: exx_energy_only, compute_kernel_fourier
    using LinearAlgebra
    using FastGaussQuadrature

    # We use kernel-specific lattice constants to bypass slow computations on huge grids
    a_vals_for = Dict(
        :sph   => [14.0, 18.0, 22.0],
        :ws    => [14.0, 18.0, 22.0],
        :probe => [14.0, 18.0, 22.0, 26.0],
        :vox   => [50.0, 62.5, 75.0, 87.5, 100.0]
    )
    
    # Pre-allocate dictionary to store evaluated EXX energies
    E_exx = Dict(name => Float64[] for name in keys(a_vals_for))

    kernels_dict = Dict(
        :probe => Coulomb(ProbeCharge()),
        :vox   => Coulomb(VoxelAveraged()),
        :sph   => SphericallyTruncatedCoulomb(),
        :ws    => WignerSeitzTruncatedCoulomb()
    )

    # Determine unique lattice constants
    unique_a_vals = sort(unique(vcat(values(a_vals_for)...)))

    # Evaluate exact exchange energies
    # We use an analytical Gaussian charge to bypass the SCF loop and egg-box effects.
    for a in unique_a_vals
        active_names = [name for (name, a_vals) in a_vals_for if a in a_vals]
        if isempty(active_names)
            continue
        end

        lattice = [a 0.0 0.0; 0.0 a 0.0; 0.0 0.0 a]
        model = Model(lattice; n_electrons=2)
        basis = PlaneWaveBasis(model; Ecut=15, kgrid=(1, 1, 1))

        kpt = basis.kpoints[1]
        ψk = zeros(ComplexF64, length(kpt.G_vectors), 1)
        
        # Model a Gaussian localized charge with real-space standard deviation of 1/\sqrt{c}
        # At Ecut=15 (Gsq_max = 30), exp(-30 / 2.0) ≈ 3e-7, guaranteeing zero truncation ringing.
        c = 2.00
        for (iG, G) in enumerate(kpt.G_vectors)
            G_cart = basis.model.recip_lattice * G
            Gsq = sum(abs2, G_cart)
            ψk[iG, 1] = exp(-Gsq / c)
        end

        ψk_real = zeros(eltype(ψk), basis.fft_size..., 1)
        @views ifft!(ψk_real[:, :, :, 1], basis, kpt, ψk[:, 1])
        ψk_real ./= sqrt(sum(abs2, ψk_real) * basis.dvol)
        occk = [2.0]

        for name in active_names
            kernel = kernels_dict[name]
            kf = compute_kernel_fourier(kernel, basis)
            E = exx_energy_only(basis, kpt, kf, ψk_real, occk)
            push!(E_exx[name], E)
        end
    end

    # Extrapolate E(a) = E_inf + c_1 / a + c_3 / a^3 + c_5 / a^5
    # The true Makov-Payne expansion for a localized density in a simple cubic cell
    # only contains odd powers. The 1/a^5 term is highly non-negligible for VoxelAveraged!
    extrapolate_poly_vox(a_vals, E_vals) = ([ones(length(a_vals)) (1 ./ a_vals) (1 ./ a_vals).^3 (1 ./ a_vals).^5] \ E_vals)[1]

    # Extrapolate E(a) = E_inf + c / a^3 (for ProbeCharge)
    extrapolate_poly_probe(a_vals, E_vals) = ([ones(length(a_vals)) (1 ./ a_vals).^3] \ E_vals)[1]
    
    # Extrapolate E(a) = E_inf + c * exp(-alpha a) for equally spaced a_vals (for SphericallyTruncated and WignerSeitz)
    function extrapolate_exp(E_vals)
        d1, d2 = E_vals[1] - E_vals[2], E_vals[2] - E_vals[3]
        abs(d1) < 1e-10 ? E_vals[3] : E_vals[3] - d2^2 / (d1 - d2)
    end

    E_inf = Dict(
        :probe => extrapolate_poly_probe(a_vals_for[:probe], E_exx[:probe]),
        :vox   => extrapolate_poly_vox(a_vals_for[:vox], E_exx[:vox]),
        :sph   => extrapolate_exp(E_exx[:sph]),
        :ws    => extrapolate_exp(E_exx[:ws])
    )

    # All extrapolated EXX energies should agree to roughly 1e-5 Ha
    E_ref = E_inf[:ws]
    for name in [:probe, :vox, :sph]
        @test abs(E_inf[name] - E_ref) < 1e-5
    end
end
