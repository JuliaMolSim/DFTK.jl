@testitem "Reference tests of Coulomb-like kernels" tags=[:exx,:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using DFTK: exx_energy_only, compute_kernel_fourier
    using .TestCases: silicon
    using LinearAlgebra

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
