using DFTK
using Test
using LinearAlgebra: norm

include("testcases.jl")

function test_chi0(;symmetry=false, use_symmetry=false, temperature=0,
                   spin_polarization=:none,
                   kgrid=[3, 1, 1], fft_size=[10, 1, 10], Ecut=3)
    testcase = silicon
    n_bands  = 12
    tol      = 1e-14
    ε        = 1e-8
    testtol  = 2e-6

    collinear = spin_polarization == :collinear
    label = [
        symmetry        ? "   symm" : "no symm",
        use_symmetry    ? "   use" : "no use",
        temperature > 0 ? "temp" : "  0K",
        collinear       ? "coll" : "none",
    ]
    @testset "Computing χ0 ($(join(label, ", ")))" begin
        spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
        magnetic_moments = collinear ? [spec => 2rand(2)] : []
        model_kwargs = (temperature=temperature, symmetries=symmetry,
                        magnetic_moments=magnetic_moments,
                        spin_polarization=spin_polarization)
        basis_kwargs = (kgrid=kgrid, fft_size=fft_size, use_symmetry=use_symmetry)
        model = model_LDA(testcase.lattice, [spec => testcase.positions]; model_kwargs...)
        basis = PlaneWaveBasis(model, Ecut; basis_kwargs...)

        ρ0     = guess_density(basis, magnetic_moments)
        energies, ham0 = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0)
        res = DFTK.next_density(ham0, tol=tol, eigensolver=diag_full, n_bands=n_bands)
        ρ1  = res.ρout

        # Now we make the same model, but add an artificial external potential ε * dV
        n_spin = model.n_spin_components
        dV = randn(eltype(basis), basis.fft_size..., n_spin)
        dV_sym = DFTK.symmetrize(basis, dV)
        if symmetry
            dV = dV_sym
        else
            @test dV_sym ≈ dV
        end

        εdV = ε * dV
        term_builder = basis -> DFTK.TermExternal(basis, εdV)
        model = model_LDA(testcase.lattice, [spec => testcase.positions];
                          model_kwargs..., extra_terms=[term_builder])
        basis = PlaneWaveBasis(model, Ecut; basis_kwargs...)
        energies, ham = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0)
        res = DFTK.next_density(ham, tol=tol, eigensolver=diag_full, n_bands=n_bands)
        ρ2     = res.ρout
        diff_findiff = (ρ2 - ρ1) / ε

        EVs = [eigen(Hermitian(Array(Hk))) for Hk in ham0.blocks]
        Es = [EV.values[1:n_bands] for EV in EVs]
        Vs = [EV.vectors[:, 1:n_bands] for EV in EVs]
        occ, εF = DFTK.compute_occupation(basis, Es)

        # Test apply_χ0 and compare against finite differences
        @test norm(diff_findiff - diff_applied_χ0) < testtol

        # just to cover it here
        if temperature > 0
            N = compute_nos(εF, basis, Es)
            D = compute_dos(εF, basis, Es)
            LDOS = compute_ldos(εF, basis, Es, Vs)
        end

        if !symmetry
            # Test compute_χ0 against finite differences
            χ0 = compute_χ0(ham0)
            diff_computed_χ0 = reshape(χ0 * vec(dV), basis.fft_size..., n_spin)
            @test norm(diff_findiff - diff_computed_χ0) < testtol

            # Test that apply_χ0 is self-adjoint
            dV1 = randn(eltype(basis), basis.fft_size..., n_spin)
            dV2 = randn(eltype(basis), basis.fft_size..., n_spin)
            χ0dV1 = apply_χ0(ham0, Vs, εF, Es, dV1)
            χ0dV2 = apply_χ0(ham0, Vs, εF, Es, dV2)
            @test abs(dot(dV1, χ0dV2) - dot(dV2, χ0dV1)) < testtol
        end
    end
end

for temperature in (0, 0.03), spin_polarization in (:none, :collinear)
    for use_symmetry in (false, true), symmetry in (false, true)
        test_chi0(;symmetry=symmetry, use_symmetry=use_symmetry,
                  temperature=temperature, spin_polarization=spin_polarization)
    end
end
