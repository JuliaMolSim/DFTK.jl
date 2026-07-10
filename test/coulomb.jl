@testitem "Reference energy tests of Coulomb kernels" tags=[:exx,:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using DFTK: exx_energy_only, eval_kernel_fourier
    using FastGaussQuadrature
    using LinearAlgebra
    using .TestCases: silicon

    # TODO: This is a bad test, better test properties, see details at the end of file

    Si = ElementPsp(14, load_psp(silicon.psp_upf))
    model  = model_DFT(silicon.lattice, [Si, Si], silicon.positions; functionals=PBE())
    basis  = PlaneWaveBasis(model; Ecut=10, kgrid=(1, 1, 1))
    scfres = self_consistent_field(basis; tol=1e-10, callback=identity)
    kf(kernel) = eval_kernel_fourier(kernel, basis)

    n_occ = 4
    kpt  = basis.kpoints[1]
    ψk   = scfres.ψ[1][:, 1:n_occ]
    occk = scfres.occupation[1][1:n_occ]
    ψk_real = similar(ψk, eltype(ψk), basis.fft_size..., n_occ)
    @views for n = 1:n_occ
        ifft!(ψk_real[:, :, :, n], basis, kpt, ψk[:, n])
    end

    k_probe = kf(ProbeCharge(BareCoulomb()))
    @testset "Coulomb with ProbeCharge" begin
        E_probe = exx_energy_only(basis, kpt, k_probe, ψk_real, occk)
        E_ref = -2.3383182267715146
        @test abs(E_ref - E_probe) < 1e-6
    end

    @testset "Coulomb with ReplaceSingularity" begin
        k_neglect = kf(ReplaceSingularity(BareCoulomb(), 0.0))
        E_neglect = exx_energy_only(basis, kpt, k_neglect, ψk_real, occk)
        E_ref = -0.7349576391651506
        @test abs(E_ref - E_neglect) < 1e-6
        @test norm(k_neglect[2:end] - k_probe[2:end]) < 1e-6
    end

    @testset "LongRangeCoulomb with ProbeCharge" begin
        k_lr = kf(ProbeCharge(LongRangeCoulomb(0.1)))
        E_lr = exx_energy_only(basis, kpt, k_lr, ψk_real, occk)
        E_ref = -0.44269774759135283
        @test abs(E_ref - E_lr) < 1e-6
    end

    @testset "ShortRangeCoulomb" begin
        k_sr = kf(ShortRangeCoulomb(0.1))
        E_sr = exx_energy_only(basis, kpt, k_sr, ψk_real, occk)
        E_ref = -5.384700394476406
        @test abs(E_ref - E_sr) < 1e-6
    end

    @testset "ShortRangeCoulomb plus LongRangeCoulomb is coulomb" begin
        k_lr  = kf(ProbeCharge(LongRangeCoulomb(0.1)))
        k_sr  = kf(ShortRangeCoulomb(0.1))
        k_sum = k_lr + k_sr

        # Note: The G=0 component does not match up, because in short-range Coulomb
        # we can take the G->0 limit exactly, but in long-range we cannot and instead
        # use the ProbeCharge regularisation. However, for finite k-points the ProbeCharge
        # regularisation for LongRangeCoulomb and Coulomb does not agree, thus we need
        # to exclude the G=0 component from the test below.
        @test maximum(abs, (k_sum - k_probe)[2:end]) < 1e-12
    end

    @testset "SphericallyTruncatedCoulomb" begin
        k_strunc = kf(SphericallyTruncatedCoulomb())
        E_strunc = exx_energy_only(basis, kpt, k_strunc, ψk_real, occk)
        E_ref = -2.360170330350145
        @test abs(E_ref - E_strunc) < 1e-6

        # TODO: Test this gives a spherically truncated function.
    end

    @testset "WignerSeitzTruncatedCoulomb" begin
        k_wstrunc = kf(WignerSeitzTruncatedCoulomb())
        E_wstrunc = exx_energy_only(basis, kpt, k_wstrunc, ψk_real, occk)
        E_ref = -2.3456932201052796
        @test abs(E_ref - E_wstrunc) < 1e-6
    end

    @testset "VoxelAverage" begin
        k_voxavg = kf(VoxelAverage(BareCoulomb()))
        E_voxavg = exx_energy_only(basis, kpt, k_voxavg, ψk_real, occk)
        E_ref = -2.249044591526691
        @test abs(E_ref - E_voxavg) < 1e-6
    end
end

@testitem "Reference kernel tests of Coulomb kernels (non-cubic lattices)" #=
        =# tags=[:exx,:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using DFTK: eval_kernel_fourier
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

    kf(b) = eval_kernel_fourier(WignerSeitzTruncatedCoulomb(), b)
    @testset "WignerSeitzTruncatedCoulomb" begin
        # TODO: These tests are not super useful, they mainly enable refactoring; they
        #       should really be replaced by tests of properties of WignerSeitzTruncatedCoulomb
        #       against SphericallyTruncatedCoulomb, for example.
        k_wstrunc = kf(basis_Pt)
        @test k_wstrunc[1:5] ≈ [289.8199039694483, 39.805749013785956, 5.580324780990978,
                                4.320510620666685, 1.71839014451522]

        k_wstrunc = kf(basis_Ga)
        @test k_wstrunc[1:5] ≈ [133.48675852141807, 12.543915639034248, 1.0201128711380303,
                                1.4993914735536529, 0.2032925438414927]

        k_wstrunc = kf(basis_Sb)
        @test k_wstrunc[1:5] ≈ [133.586993467607, 17.87558501104982, 3.056073383974545,
                                1.6670099918317942, 0.9065961832826208] atol=1e-6

        k_wstrunc = kf(basis_illcond)
        @test k_wstrunc[1:5] ≈ [0.6835202703428708, 0.003317275153646196, 8.838678696668954e-5,
                                0.00024225236624065088, 5.8059844858737624e-5] atol=1e-6
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
    kf(kernel) = eval_kernel_fourier(kernel, basis)

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

        for kernel in (BareCoulomb(), LongRangeCoulomb())
            # Compute Brillouin integral of k(q) times probe charge in Fourier (e^(-α q^2))
            # We use that the integrand is spherically symmetric and thus directly introduce
            # a 4π q^2 factor
            qmax = sqrt(316 / α)  # At that point e^(-α q^2) is numerically zero
            numerical_integral, _ = quadgk(0, qmax) do q
                4π * q^2 * DFTK.eval_kernel_fourier(kernel, [q, 0, 0]) * exp(-α * q^2)
            end
            analytical_integral = DFTK.compute_probe_charge_integral(kernel, α)
            @test abs(numerical_integral - analytical_integral) < 1e-12
        end
    end
end

@testitem "Asymptotic consistency of interaction kernels for localized density" tags=[:exx,:dont_test_mpi] begin
    using DFTK
    using DFTK: exx_energy_only, eval_kernel_fourier
    using LinearAlgebra
    using FastGaussQuadrature

    # Evaluate exact exchange energies
    # We use an analytical Gaussian charge to bypass the SCF loop and egg-box effects.
    function evaluate_kernel_on_gaussian_charge(kernel, a; Ecut=15, c=2.0)
        lattice = diagm([a, a, a])
        model   = Model(lattice; n_electrons=2)
        basis   = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
        kpt = basis.kpoints[1]
        
        # Model a Gaussian localized charge with real-space standard deviation of 1/\sqrt{c}
        # At Ecut=15 (Gsq_max = 30), exp(-30 / 2.0) ≈ 3e-7, guaranteeing zero truncation ringing.
        ψk = zeros(ComplexF64, length(kpt.G_vectors), 1)
        ψk_real = zeros(ComplexF64, basis.fft_size..., 1)
        for (iG, G) in enumerate(kpt.G_vectors)
            G_cart = basis.model.recip_lattice * G
            Gsq = sum(abs2, G_cart)
            ψk[iG, 1] = exp(-Gsq / c)
        end
        @views ifft!(ψk_real[:, :, :, 1], basis, kpt, ψk[:, 1])
        ψk_real ./= sqrt(sum(abs2, ψk_real) * basis.dvol)  # normalise ψk
        occk = [2.0]

        kernel_values = eval_kernel_fourier(kernel, basis)
        exx_energy_only(basis, kpt, kernel_values, ψk_real, occk)
    end

    # Extrapolate E(a) = E_inf + c * exp(-alpha a) for equally spaced a_vals
    # (for SphericallyTruncated and WignerSeitz)
    function extrapolate_exp(a_vals, E_vals)
        @assert length(a_vals) == 3
        @assert a_vals[1] - a_vals[2] == a_vals[2] - a_vals[3]
        d1, d2 = E_vals[1] - E_vals[2], E_vals[2] - E_vals[3]
        abs(d1) < 1e-10 ? E_vals[3] : E_vals[3] - d2^2 / (d1 - d2)
    end

    # This is our reference value, against which we test all others
    E_WignerSeitz = let a_vals = [14.0, 18.0, 22.0]
        kernel = WignerSeitzTruncatedCoulomb()
        E_vals = evaluate_kernel_on_gaussian_charge.(Ref(kernel), a_vals)
        E_inf  = extrapolate_exp(a_vals, E_vals)
    end

    @testset "WignerSeitz against SphericallyTruncated" begin
        a_vals = [14.0, 18.0, 22.0]
        kernel = SphericallyTruncatedCoulomb()
        E_vals = evaluate_kernel_on_gaussian_charge.(Ref(kernel), a_vals)
        E_inf  = extrapolate_exp(a_vals, E_vals)

        @test abs(E_inf - E_WignerSeitz) < 1e-5
    end

    @testset "WignerSeitz against ProbeCharge" begin
        a_vals = [14.0, 18.0, 22.0, 26.0]
        kernel = ProbeCharge(BareCoulomb())
        E_vals = evaluate_kernel_on_gaussian_charge.(Ref(kernel), a_vals)

        # Extrapolate E(a) = E_inf + c / a^3 (for ProbeCharge)
        V = [ones(length(a_vals))   (1 ./ a_vals).^3]
        c = V \ E_vals  # Polynomial fit
        E_inf = c[1]    # Constant coefficient
    end

    @testset "WignerSeitz against VoxelAverage" begin
        a_vals = [50.0, 62.5, 75.0, 87.5, 100.0]
        kernel = VoxelAverage(BareCoulomb())
        E_vals = evaluate_kernel_on_gaussian_charge.(Ref(kernel), a_vals)

        # Extrapolate E(a) = E_inf + c_1 / a + c_3 / a^3 + c_5 / a^5 (for VoxelAverage)
        # We use a multipole expansion (Makov-Payne style) of a localized density.
        # Note that the 1/a^5 term is important!
        V = [ones(length(a_vals))   (1 ./ a_vals)   (1 ./ a_vals).^3   (1 ./ a_vals).^5]
        c = V \ E_vals  # Polynomial fit
        E_inf = c[1]    # Constant coefficient

        @test abs(E_inf - E_WignerSeitz) < 1e-5
    end
end


# TODO: Tests ot include in the future are
#       - Test for type stability (i.e. when Float32 is used)
