using Test
using LinearAlgebra
using DFTK
include("testcases.jl")

# TODO Once we have converged SCF densities in a file it would be better to instead / also
#      test the energies of these densities and compare them directly to the reference
#      energies obtained in the data files

@testset "Using BZ symmetry yields identical density" begin
    function get_bands(testcase, kcoords, ksymops, atoms; Ecut=5, tol=1e-8)
        kwargs = ()
        n_bands = div(testcase.n_electrons, 2)
        if testcase.Tsmear !== nothing
            kwargs = (temperature=testcase.Tsmear, smearing=DFTK.Smearing.FermiDirac())
            n_bands = div(testcase.n_electrons, 2) + 3
        end

        model = model_DFT(testcase.lattice, atoms, :lda_xc_teter93; kwargs...)
        basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
        ham = Hamiltonian(basis; ρ=guess_density(basis, atoms))

        res = diagonalize_all_kblocks(lobpcg_hyper, ham, n_bands; tol=tol)
        occ, εF = DFTK.find_occupation(basis, res.λ)
        ρnew = compute_density(basis, res.X, occ)

        basis, res.X, res.λ, ρnew.fourier
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
                                      kshift=[0, 0, 0])
        spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
        atoms = [spec => testcase.positions]

        kfull, sym_full = bzmesh_uniform(kgrid_size, kshift=kshift)
        res = get_bands(testcase, kfull, sym_full, atoms;
                        Ecut=Ecut, tol=tol)
        basis_full, ψ_full, eigenvalues_full, ρ_full = res
        test_orthonormality(basis_full, ψ_full, tol=tol)

        kcoords, ksymops = bzmesh_ir_wedge(kgrid_size, testcase.lattice, atoms, kshift=kshift)
        res = get_bands(testcase, kcoords, ksymops, atoms; Ecut=Ecut, tol=tol)
        basis_ir, ψ_ir, eigenvalues_ir, ρ_ir = res

        test_orthonormality(basis_ir, ψ_ir, tol=tol)

        # Test density is equivalent
        @test maximum(abs.(ρ_ir - ρ_full)) < 10tol

        # Test equivalent k-Points have the same orbital energies
        for (ik, k) in enumerate(kcoords)
            for (S, τ) in ksymops[ik]
                ikfull = findfirst(1:length(kfull)) do idx
                    all(isinteger, kfull[idx] - S * k)
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
    end

    test_full_vs_irreducible(silicon, [3, 3, 3], Ecut=5, tol=1e-6)
    test_full_vs_irreducible(silicon, [3, 3, 3], Ecut=5, tol=1e-6, kshift=[1/2, 1/2, 1/2])
    test_full_vs_irreducible(silicon, [3, 3, 3], Ecut=5, tol=1e-6, kshift=[1/2, 0, 1/2])
    test_full_vs_irreducible(silicon, [3, 3, 3], Ecut=5, tol=1e-6, kshift=[0, 1/2, 0])

    test_full_vs_irreducible(silicon, [2, 3, 4], Ecut=5, tol=1e-6)
    test_full_vs_irreducible(magnesium, [2, 3, 4], Ecut=5, tol=1e-6, n_ignore=1)
    test_full_vs_irreducible(aluminium, [1, 3, 5], Ecut=3, tol=1e-5, n_ignore=3)
    #
    # That's pretty expensive:
    # test_full_vs_irreducible([4, 4, 4], Ecut=5, tol=1e-6)
end
