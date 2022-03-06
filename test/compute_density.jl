using Test
using LinearAlgebra
using DFTK
import DFTK: total_local_potential
include("testcases.jl")

# TODO Once we have converged SCF densities in a file it would be better to instead / also
#      test the energies of these densities and compare them directly to the reference
#      energies obtained in the data files

if mpi_nprocs() == 1  # not easy to distribute
@testset "Using BZ symmetry yields identical density" begin
    function get_bands(testcase, kcoords, kweights, symmetries; Ecut=5, tol=1e-8, n_rounds=1)
        kwargs = ()
        n_bands = div(testcase.n_electrons, 2, RoundUp)
        if testcase.temperature !== nothing
            kwargs = (temperature=testcase.temperature, smearing=DFTK.Smearing.FermiDirac())
            n_bands = div(testcase.n_electrons, 2, RoundUp) + 4
        end

        model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions,
                          :lda_xc_teter93; kwargs...)
        basis = PlaneWaveBasis(model, Ecut, kcoords, kweights, symmetries)
        ham = Hamiltonian(basis; ρ=guess_density(basis, testcase.atoms, testcase.positions))

        res = diagonalize_all_kblocks(lobpcg_hyper, ham, n_bands; tol)
        occ, εF = DFTK.compute_occupation(basis, res.λ)
        ρnew = compute_density(basis, res.X, occ)

        for it in 1:n_rounds
            ham = Hamiltonian(basis; ρ=ρnew)
            res = diagonalize_all_kblocks(lobpcg_hyper, ham, n_bands; tol=tol, ψguess=res.X)

            occ, εF = DFTK.compute_occupation(basis, res.λ)
            ρnew = compute_density(basis, res.X, occ)
        end

        ham, res.X, res.λ, ρnew, occ
    end

    function test_orthonormality(basis, ψ; tol=1e-8)
        n_k = length(ψ)
        n_states = size(ψ[1], 2)
        n_fft = prod(basis.fft_size)

        for (ik, kpt) in enumerate(basis.kpoints)
            # Fourier-transform the wave functions to real space
            ψk = ψ[ik]
            ψk_real = cat((DFTK.G_to_r(basis, kpt, ψik) for ψik in eachcol(ψk))..., dims=4)

            T = real(eltype(ψk_real))
            ψk_real_mat = reshape(ψk_real, n_fft, n_states)
            ψk_real_overlap = adjoint(ψk_real_mat) * ψk_real_mat
            nondiag = ψk_real_overlap - I * (n_fft / basis.model.unit_cell_volume)

            @test maximum(abs.(nondiag)) < tol
        end
    end

    function test_full_vs_irreducible(testcase, kgrid_size; Ecut=5, tol=1e-8, n_ignore=0,
                                      kshift=[0, 0, 0], eigenvectors=true)
        kfull, kwfull, symmetries = bzmesh_uniform(kgrid_size, kshift)
        res = get_bands(testcase, kfull, kwfull, symmetries; Ecut, tol)
        ham_full, ψ_full, eigenvalues_full, ρ_full, occ_full = res
        test_orthonormality(ham_full.basis, ψ_full; tol)

        symmetries = DFTK.symmetry_operations(testcase.lattice, testcase.atoms,
                                              testcase.positions)
        kcoords, kweights, symmetries = bzmesh_ir_wedge(kgrid_size, symmetries, kshift)
        res = get_bands(testcase, kcoords, kweights, symmetries; Ecut, tol)
        ham_ir, ψ_ir, eigenvalues_ir, ρ_ir, occ_ir = res
        test_orthonormality(ham_ir.basis, ψ_ir; tol)
        @test ham_full.basis.fft_size == ham_ir.basis.fft_size

        # Test density is the same in both schemes, and symmetric wrt the basis symmetries
        @test maximum(abs.(ρ_ir - ρ_full)) < 10tol
        @test maximum(abs, DFTK.symmetrize_ρ(ham_ir.basis, ρ_ir; symmetries) - ρ_ir) < tol

        # Test local potential is the same in both schemes
        @test maximum(abs, total_local_potential(ham_ir) - total_local_potential(ham_full)) < tol

        # Test equivalent k-points have the same orbital energies
        for (ik, k) in enumerate(kcoords)
            for symop in symmetries
                ikfull = findfirst(1:length(kfull)) do idx
                    all(isinteger, kfull[idx] - symop.S * k)
                end
                @test ikfull !== nothing

                # Orbital energies should agree (by symmetry)
                # The largest few are ignored, because the results between the
                # full kpoints and the reduced kpoints are sometimes a little unstable
                # (since we actually use different guesses)
                range = 1:length(eigenvalues_full[ikfull]) - n_ignore
                @test eigenvalues_full[ikfull][range] ≈ eigenvalues_ir[ik][range] atol=tol
            end
        end

        if eigenvectors
            # Test applying the symmetry transformation to the irreducible k-points
            # yields an eigenfunction of the Hamiltonian
            # Also check that the accumulated partial densities are equal
            # to the returned density.
            ρsum = zeros(eltype(ψ_ir[1]), ham_ir.basis.fft_size)
            n_ρ = 0
            for (ik, k) in enumerate(kcoords)
                Hk_ir = ham_ir.blocks[ik]
                for symop in symmetries
                    Skpoint, ψSk = DFTK.apply_symop(symop, ham_ir.basis, Hk_ir.kpoint, ψ_ir[ik])

                    ikfull = findfirst(1:length(kfull)) do idx
                        all(isinteger, round.(kfull[idx] - Skpoint.coordinate, digits=10))
                    end
                    @test !isnothing(ikfull)
                    Hk_full = ham_full.blocks[ikfull]

                    range = 1:length(eigenvalues_ir[ik]) - n_ignore
                    for iband in range
                        ψnSk = ψSk[:, iband]
                        residual = norm(Hk_full * ψnSk - eigenvalues_ir[ik][iband] * ψnSk)
                        @test residual < 10tol
                    end  # iband
                end  # symop
            end  # k
        end # eigenvectors
    end

    test_full_vs_irreducible(silicon, [3, 2, 3], Ecut=5, tol=1e-10)

    test_full_vs_irreducible(silicon, [3, 3, 3], Ecut=5, tol=1e-6)
    test_full_vs_irreducible(silicon, [3, 3, 3], Ecut=5, tol=1e-6, kshift=[1/2, 1/2, 1/2])
    test_full_vs_irreducible(silicon, [3, 3, 3], Ecut=5, tol=1e-6, kshift=[1/2, 0, 1/2])
    test_full_vs_irreducible(silicon, [3, 3, 3], Ecut=5, tol=1e-6, kshift=[0, 1/2, 0])

    test_full_vs_irreducible(silicon, [2, 3, 4], Ecut=5, tol=1e-6)
    test_full_vs_irreducible(magnesium, [2, 3, 4], Ecut=5, tol=1e-6, n_ignore=1)
    test_full_vs_irreducible(aluminium, [1, 3, 5], Ecut=3, tol=1e-5, n_ignore=3,
                             eigenvectors=false)
end
end
