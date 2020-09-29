using DFTK
using Test
using LinearAlgebra: norm

include("testcases.jl")

@testset "Computing χ0" begin
    Ecut=3
    fft_size = [10, 1, 10]
    tol=1e-14
    ε = 1e-8
    testtol = 2e-6
    kgrid = [3, 1, 1]
    testcase = silicon
    n_bands = 12

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    for temperature in (0, 0.03)
        for symmetry in (false, true)
            for use_symmetry in (false, true)
                model = model_LDA(testcase.lattice, [spec => testcase.positions],
                                  temperature=temperature, symmetry=(symmetry ? :force : :off))
                basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=fft_size,
                                       use_symmetry=use_symmetry)

                ρ0 = guess_density(basis)
                energies, ham0 = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0, ρspin=nothing)
                ρ1 = DFTK.next_density(ham0, tol=tol, eigensolver=diag_full, n_bands=n_bands).ρout

                # Now we make the same model, but add an artificial external potential ε * dV
                dV = from_real(basis, randn(eltype(basis), basis.fft_size))
                dV_sym = DFTK.symmetrize(dV)
                if symmetry
                    dV = dV_sym
                else
                    @test dV_sym.real ≈ dV.real
                end

                term_builder = basis -> DFTK.TermExternal(basis, ε .* dV.real)
                model = model_LDA(testcase.lattice, [spec => testcase.positions],
                                  temperature=temperature, extra_terms=[term_builder],
                                  symmetry=(symmetry ? :force : :off))
                basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=fft_size,
                                       use_symmetry=use_symmetry)
                energies, ham = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0, ρspin=nothing)
                ρ2 = DFTK.next_density(ham, tol=tol, eigensolver=diag_full, n_bands=n_bands).ρout
                diff_findiff = (ρ2 - ρ1) / ε


                EVs = [eigen(Hermitian(Array(Hk))) for Hk in ham0.blocks]
                Es = [EV.values[1:n_bands] for EV in EVs]
                Vs = [EV.vectors[:, 1:n_bands] for EV in EVs]
                occ, εF = find_occupation(basis, Es)

                # Test apply_χ0
                diff_applied_χ0 = apply_χ0(ham0, Vs, εF, Es, dV; droptol=0)
                @test norm(diff_findiff.real - diff_applied_χ0.real) < testtol

                # Test compute_χ0 against finite differences
                if !symmetry
                    χ0 = compute_χ0(ham0)
                    diff_computed_χ0 = real(reshape(χ0 * vec(dV.real), basis.fft_size))
                    @test norm(diff_findiff.real - diff_computed_χ0) < testtol

                    # Test that apply_χ0 is self-adjoint
                    dV1 = from_real(basis, randn(eltype(basis), basis.fft_size))
                    dV2 = from_real(basis, randn(eltype(basis), basis.fft_size))
                    χ0dV1 = apply_χ0(ham0, Vs, εF, Es, dV1)
                    χ0dV2 = apply_χ0(ham0, Vs, εF, Es, dV2)
                    @test abs(dot(dV1.real, χ0dV2.real) - dot(dV2.real, χ0dV1.real)) < testtol

                    # Test the diagonal_only option
                    χ0_diag = compute_χ0(ham0; droptol=Inf)
                    diff_diag_1 = real(reshape(χ0_diag * vec(dV.real), basis.fft_size))
                    diff_diag_2 = apply_χ0(ham0, Vs, εF, Es, dV; droptol=Inf,
                                           sternheimer_contribution=false)
                    @test norm(diff_diag_1 - diff_diag_2.real) < testtol
                end
            end
        end
    end
end
