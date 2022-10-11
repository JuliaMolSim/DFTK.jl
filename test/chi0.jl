using DFTK
import DFTK: mpi_mean!
using MPI
using Test
using LinearAlgebra: norm

include("testcases.jl")

function test_chi0(testcase; symmetries=false, temperature=0, spin_polarization=:none,
                   eigensolver=lobpcg_hyper, Ecut=10, kgrid=[3, 1, 1], fft_size=[15, 1, 15],
                   compute_full_χ0=false, εF=nothing)

    tol      = 1e-11
    ε        = 1e-6
    testtol  = 2e-6

    collinear   = spin_polarization == :collinear
    is_metal    = !isnothing(testcase.temperature)
    is_εF_fixed = !isnothing(εF)
    eigsol = eigensolver == lobpcg_hyper
    label = [
        is_metal        ? "    metal" : "insulator",
        eigsol          ? "   lobpcg" : "full diag",
        symmetries      ? "   symm" : "no symm",
        temperature > 0 ? "temp" : "  0K",
        collinear       ? "coll" : "none",
        is_εF_fixed     ? "NVT" : "μVT",
    ]
    @testset "Computing χ0 ($(join(label, ", ")))" begin
        magnetic_moments = collinear ? [0.3, 0.7] : []
        model_kwargs = (; symmetries, magnetic_moments, spin_polarization, temperature, εF,
                        force_electrostatics=true)
        basis_kwargs = (; kgrid, fft_size, Ecut)

        model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                          model_kwargs...)
        basis = PlaneWaveBasis(model; basis_kwargs...)
        ρ0 = guess_density(basis, magnetic_moments)
        _, ham0   = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0)
        nbandsalg = is_εF_fixed ? FixedBands(; n_bands_converge=6) : AdaptiveBands(model)
        res = DFTK.next_density(ham0, nbandsalg; tol, eigensolver)
        scfres = (ham=ham0, res...)

        # create external small perturbation εδV
        n_spin = model.n_spin_components
        δV = randn(eltype(basis), basis.fft_size..., n_spin)
        mpi_mean!(δV, MPI.COMM_WORLD)
        δV_sym = DFTK.symmetrize_ρ(basis, δV, symmetries=model.symmetries)
        if symmetries
            δV = δV_sym
        else
            @test δV_sym ≈ δV
        end

        function compute_ρ_FD(ε)
            term_builder = basis -> DFTK.TermExternal(ε * δV)
            model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                              model_kwargs..., extra_terms=[term_builder])
            basis = PlaneWaveBasis(model; basis_kwargs...)
            _, ham = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0)
            res = DFTK.next_density(ham, nbandsalg; tol, eigensolver)
            res.ρout
        end

        # middle point finite difference for more precision
        ρ1 = compute_ρ_FD(-ε)
        ρ2 = compute_ρ_FD(ε)
        diff_findiff = (ρ2 - ρ1) / (2ε)

        # Test apply_χ0 and compare against finite differences
        diff_applied_χ0 = apply_χ0(scfres, δV)
        @test norm(diff_findiff - diff_applied_χ0) < testtol

        # Test apply_χ0 without extra bands
        ψ_occ, occ_occ = DFTK.select_occupied_orbitals(basis,
                                                       scfres.ψ,
                                                       scfres.occupation;
                                                       threshold=scfres.occupation_threshold)
        ε_occ = [scfres.eigenvalues[ik][1:size(ψk, 2)] for (ik, ψk) in enumerate(ψ_occ)]

        diff_applied_χ0_noextra = apply_χ0(scfres.ham, ψ_occ, occ_occ, scfres.εF,
                                           ε_occ, δV; scfres.occupation_threshold)
        @test norm(diff_applied_χ0_noextra - diff_applied_χ0) < testtol

        # just to cover it here
        if temperature > 0
            D = compute_dos(res.εF, basis, res.eigenvalues)
            LDOS = compute_ldos(res.εF, basis, res.eigenvalues, res.ψ)
        end

        if !symmetries
            #  Test compute_χ0 against finite differences
            #  (only works in reasonable time for small Ecut)
            if compute_full_χ0
                χ0 = compute_χ0(ham0)
                diff_computed_χ0 = reshape(χ0 * vec(δV), basis.fft_size..., n_spin)
                @test norm(diff_findiff - diff_computed_χ0) < testtol
            end

            # Test that apply_χ0 is self-adjoint
            δV1 = randn(eltype(basis), basis.fft_size..., n_spin)
            δV2 = randn(eltype(basis), basis.fft_size..., n_spin)
            mpi_mean!(δV1, MPI.COMM_WORLD)
            mpi_mean!(δV2, MPI.COMM_WORLD)

            χ0δV1 = apply_χ0(scfres, δV1)
            χ0δV2 = apply_χ0(scfres, δV2)
            @test abs(dot(δV1, χ0δV2) - dot(δV2, χ0δV1)) < testtol
        end
    end
end

@testset "Computing χ0" begin
    for (case, temperatures) in [(silicon, (0, 0.03)), (magnesium, (0.01, ))]
        for temperature in temperatures, spin_polarization in (:none, :collinear)
            for symmetries in (false, true)
                test_chi0(case; symmetries, temperature, spin_polarization)
            end
        end
    end

    # Additional test for compute_χ0
    for spin_polarization in (:none, :collinear)
        test_chi0(silicon; symmetries=false, spin_polarization, eigensolver=diag_full,
                  Ecut=3, fft_size=[10, 1, 10], compute_full_χ0=true)
        test_chi0(magnesium; spin_polarization, temperature=0.01, εF=0.3)
    end
end
