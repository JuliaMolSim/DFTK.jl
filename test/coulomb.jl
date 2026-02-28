@testitem "Reference tests for exx implementations" tags=[:exx,:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using DFTK: exx_energy_only, compute_kernel_fourier
    using .TestCases: silicon
    using LinearAlgebra
    using QuadGK

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

    k_probe = compute_kernel_fourier(basis; interaction_kernel=Coulomb(ProbeCharge()))
    @testset "Coulomb with ProbeCharge" begin
        E_probe = exx_energy_only(basis, kpt, k_probe, ψk_real, occk)
        E_ref = -2.3383063575660987
        @test abs(E_ref - E_probe) < 1e-6
    end

    @testset "Coulomb with ReplaceSingularity" begin
        k_neglect = compute_kernel_fourier(basis; interaction_kernel=Coulomb(ReplaceSingularity()))
        E_neglect = exx_energy_only(basis, kpt, k_neglect, ψk_real, occk)
        E_ref = -0.7349457693125514
        @test abs(E_ref - E_neglect) < 1e-6
        @test norm(k_neglect[2:end] - k_probe[2:end]) < 1e-6
    end

    @testset "ErfLongRangeCoulomb with ProbeCharge" begin
        k_lr = compute_kernel_fourier(basis; interaction_kernel=ErfLongRangeCoulomb(0.1, ProbeCharge()))
        E_lr = exx_energy_only(basis, kpt, k_lr, ψk_real, occk)
        E_ref = -0.44269774759135383
        @test abs(E_ref - E_lr) < 1e-6
    end

    @testset "ErfShortRangeCoulomb" begin
        k_sr = compute_kernel_fourier(basis; interaction_kernel=ErfShortRangeCoulomb(0.1))
        E_sr = exx_energy_only(basis, kpt, k_sr, ψk_real, occk)
        E_ref = -5.384688524633953
        @test abs(E_ref - E_sr) < 1e-6
    end

    @testset "SphericallyTruncatedCoulomb" begin
        k_strunc = compute_kernel_fourier(basis; interaction_kernel=SphericallyTruncatedCoulomb())
        E_strunc = exx_energy_only(basis, kpt, k_strunc, ψk_real, occk)
        E_ref = -2.360166200435632
        @test abs(E_ref - E_strunc) < 1e-6

        # TODO: Test this gives a spherically truncated function.
    end

    @testset "Probe-charge integrals" begin
        α = π^2 / basis.Ecut
        Ω = basis.model.unit_cell_volume

        # we want to compute ∫ f(G) from 0 to Gmax
        Gmax = 4*sqrt(2*basis.Ecut)
        f(kernel, Gsq) = DFTK.eval_kernel_fourier(kernel, Gsq) * Gsq * exp(-α*Gsq)

        # loop through all kernels with evaluate_probe_charge_integral
        for kernel in (Coulomb(), ErfLongRangeCoulomb())
            numerical_integral, _ = quadgk(G -> f(kernel, G^2) , 0, Gmax)
            numerical_integral *= Ω/(2π)^3 * 4π  # multipliy FFT factor and solid angle 
            analytical_integral = DFTK.evaluate_probe_charge_integral(kernel, α, Ω)

            @test abs(numerical_integral - analytical_integral) < 1e-12
        end
    end
end

# TODO: Tests ot include in the future are
#       - Test for type stability (i.e. when Float32 is used)
#       - Test properties, such as the fact that
#         the Coulomb energy converges to the same value as the supercell / k-grid
#         more suprecells are used
