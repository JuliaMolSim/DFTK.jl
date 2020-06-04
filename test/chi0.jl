using DFTK
using Test
using LinearAlgebra: norm

include("testcases.jl")

@testset "Computing χ0" begin
    Ecut=3
    fft_size = [10, 1, 10]
    tol=1e-14
    ε = 1e-8
    testtol = 1e-6
    kgrid = [3, 1, 1]
    testcase = silicon
    n_bands = 12

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    for temperature in (0, 0.03)
        for symmetry in (false, true)
            model = model_LDA(testcase.lattice, [spec => testcase.positions], temperature=temperature, symmetry=(symmetry ? :force : :off))
            basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=fft_size)

            ρ0 = guess_density(basis)
            energies, ham0 = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0)
            ρ1 = DFTK.next_density(ham0, tol=tol, eigensolver=diag_full, n_bands=n_bands).ρ

            # Now we make the same model, but add an artificial external potential ε * dV
            dV = randn(eltype(basis), basis.fft_size)
            dV_sym = DFTK.symmetrize(from_real(basis, dV)).real
            if symmetry
                dV = dV_sym
            else
                @test dV_sym ≈ dV
            end

            term_builder = basis -> DFTK.TermExternal(basis, ε .* dV)
            model = model_LDA(testcase.lattice, [spec => testcase.positions], temperature=temperature, extra_terms=[term_builder], symmetry=(symmetry ? :force : :off))
            basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=fft_size)
            energies, ham = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0)
            ρ2 = DFTK.next_density(ham, tol=tol, eigensolver=diag_full, n_bands=n_bands).ρ
            diff_findiff = (ρ2.real - ρ1.real)/ε


            EVs = [eigen(Hermitian(Array(Hk))) for Hk in ham0.blocks]
            Es = [EV.values[1:n_bands] for EV in EVs]
            Vs = [EV.vectors[:, 1:n_bands] for EV in EVs]
            occ, εF = find_occupation(basis, Es)

            # Test apply_χ0
            diff_applied_χ0 = apply_χ0(ham0, dV, Vs, εF, Es; droptol=0)
            @test norm(diff_findiff - diff_applied_χ0) < testtol

            # Test compute_χ0 against finite differences
            if !symmetry
                χ0 = compute_χ0(ham0)
                diff_computed_χ0 = real(reshape(χ0*vec(dV), basis.fft_size))
                @test norm(diff_findiff - diff_computed_χ0) < testtol

                # Test that apply_χ0 is self-adjoint
                dV1 = randn(eltype(basis), basis.fft_size)
                dV2 = randn(eltype(basis), basis.fft_size)
                χ0dV1 = apply_χ0(ham0, dV1, Vs, εF, Es)
                χ0dV2 = apply_χ0(ham0, dV2, Vs, εF, Es)
                @test abs(dot(dV1, χ0dV2) - dot(dV2, χ0dV1)) < testtol

                # Test the diagonal_only option
                χ0_diag = compute_χ0(ham0; droptol=Inf)
                diff_diag_1 = real(reshape(χ0_diag*vec(dV), basis.fft_size))
                diff_diag_2 = apply_χ0(ham0, dV, Vs, εF, Es; droptol=Inf,
                                       sternheimer_contribution=false)
                @test norm(diff_diag_1 - diff_diag_2) < testtol
            end
        end
    end
end
