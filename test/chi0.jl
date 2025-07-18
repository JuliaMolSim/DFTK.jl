@testmodule Chi0 begin
using Test
using DFTK
using DFTK: mpi_mean!
using MPI
using LinearAlgebra

function test_chi0(testcase; symmetries=false, temperature=0, spin_polarization=:none,
                   eigensolver=lobpcg_hyper, Ecut=10, kgrid=[3, 1, 1], fft_size=[15, 1, 15],
                   compute_full_χ0=false, εF=nothing, functionals=LDA(), tol=1e-11,
                   ε=1e-6, atol=2e-6)

    collinear   = spin_polarization == :collinear
    is_εF_fixed = !isnothing(εF)
    eigsol = eigensolver == lobpcg_hyper
    label = [
        testcase.is_metal ? "    metal" : "insulator",
        eigsol            ? "   lobpcg" : "full diag",
        symmetries        ? "   symm" : "no symm",
        temperature > 0   ? "temp" : "  0K",
        collinear         ? "coll" : "none",
        is_εF_fixed       ? "  εF" : "none",
    ]
    @testset "Computing χ0 ($(join(label, ", ")))" begin
        magnetic_moments = collinear ? [0.3, 0.7] : []
        model_kwargs = (; symmetries, magnetic_moments, spin_polarization, temperature, εF,
                        disable_electrostatics_check=true)
        basis_kwargs = (; kgrid, fft_size, Ecut)

        model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                          functionals, model_kwargs...)
        basis = PlaneWaveBasis(model; basis_kwargs...)
        ρ0    = guess_density(basis, magnetic_moments)
        ham0  = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0).ham
        nbandsalg = is_εF_fixed ? FixedBands(; n_bands_converge=6) : AdaptiveBands(model)
        res = DFTK.next_density(ham0, nbandsalg; tol, eigensolver)
        scfres = (; ham=ham0, res...)

        # create external small perturbation εδV
        n_spin = model.n_spin_components
        δV = randn(eltype(basis), basis.fft_size..., n_spin)
        mpi_mean!(δV, MPI.COMM_WORLD)
        δV_sym = DFTK.symmetrize_ρ(basis, δV; model.symmetries)
        if symmetries
            δV = δV_sym
        else
            @test δV_sym ≈ δV
        end

        function compute_ρ_FD(ε)
            term_builder = basis -> DFTK.TermExternal(ε * δV)
            model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                              functionals, model_kwargs..., extra_terms=[term_builder])
            basis = PlaneWaveBasis(model; basis_kwargs...)
            ham = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0).ham
            res = DFTK.next_density(ham, nbandsalg; tol, eigensolver)
            res.ρout
        end

        # middle point finite difference for more precision
        ρ1 = compute_ρ_FD(-ε)
        ρ2 = compute_ρ_FD(ε)
        diff_findiff = (ρ2 - ρ1) / (2ε)

        # Test apply_χ0 and compare against finite differences
        diff_applied_χ0 = apply_χ0(scfres, δV).δρ
        @test norm(diff_findiff - diff_applied_χ0) < atol

        # Test apply_χ0 without extra bands
        ψ_occ, occ_occ = DFTK.select_occupied_orbitals(basis, scfres.ψ, scfres.occupation;
                                                       threshold=scfres.occupation_threshold)
        ε_occ = [scfres.eigenvalues[ik][1:size(ψk, 2)] for (ik, ψk) in enumerate(ψ_occ)]

        diff_applied_χ0_noextra = apply_χ0(scfres.ham, ψ_occ, occ_occ, scfres.εF, ε_occ, δV;
                                           scfres.occupation_threshold).δρ
        @test norm(diff_applied_χ0_noextra - diff_applied_χ0) < atol

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
                @test norm(diff_findiff - diff_computed_χ0) < atol
            end

            # Test that apply_χ0 is self-adjoint
            δV1 = randn(eltype(basis), basis.fft_size..., n_spin)
            δV2 = randn(eltype(basis), basis.fft_size..., n_spin)
            mpi_mean!(δV1, MPI.COMM_WORLD)
            mpi_mean!(δV2, MPI.COMM_WORLD)

            χ0δV1 = apply_χ0(scfres, δV1).δρ
            χ0δV2 = apply_χ0(scfres, δV2).δρ
            @test abs(dot(δV1, χ0δV2) - dot(δV2, χ0δV1)) < atol
        end
    end
end
end

@testitem "Computing χ0" setup=[Chi0, TestCases] begin
    using DFTK
    using .Chi0: test_chi0
    (; silicon, magnesium) = TestCases.all_testcases

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

@testitem "Apply χ0 for large band gap" setup=[Chi0, TestCases] begin
    using DFTK
    using PseudoPotentialData
    using .Chi0: test_chi0

    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_5.standard.upf")
    atoms = [ElementPsp(:Na, pseudopotentials),
             ElementPsp(:Cl, pseudopotentials)]
    positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    lattice = (10.92 / 2) * [0 1 1; 1 0 1; 1 1 0]

    sodium_chloride = (; lattice, positions, atoms, is_metal=false)
    test_chi0(sodium_chloride;
              symmetries=true, temperature=1e-4, Ecut=20,
              kgrid=[2, 2, 2], fft_size=[36, 36, 36], atol=5e-6)
end
