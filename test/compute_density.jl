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

        model = model_dft(testcase.lattice, :lda_xc_teter93, atoms; kwargs...)
        basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)
        ham = Hamiltonian(basis, guess_density(basis, atoms))

        res = diagonalise_all_kblocks(lobpcg_hyper, ham, n_bands; tol=tol)
        occ, εF = DFTK.find_occupation(basis, res.λ)
        ρnew = compute_density(basis, res.X, occ)

        basis, res.X, res.λ, ρnew.fourier
    end

    function test_orthonormality(basis, Psi; tol=1e-8)
        n_k = length(Psi)
        n_states = size(Psi[1], 2)
        n_fft = prod(basis.fft_size)

        for (ik, kpt) in enumerate(basis.kpoints)
            # Fourier-transform the wave functions to real space
            Ψk = Psi[ik]
            Ψk_real = cat((DFTK.G_to_r(basis, kpt, Ψik) for Ψik in eachcol(Ψk))..., dims=4)

            T = real(eltype(Ψk_real))
            Ψk_real_mat = reshape(Ψk_real, n_fft, n_states)
            Ψk_real_overlap = adjoint(Ψk_real_mat) * Ψk_real_mat
            nondiag = Ψk_real_overlap - I * (n_fft / basis.model.unit_cell_volume)

            @test maximum(abs.(nondiag)) < tol
        end
    end

    function test_full_vs_irreducible(testcase, kgrid_size; Ecut=5, tol=1e-8, n_ignore=0)
        spec = Element(testcase.atnum, psp=load_psp(testcase.psp))
        atoms = (spec => testcase.positions, )

        kfull, sym_full = bzmesh_uniform(kgrid_size)
        res = get_bands(testcase, kfull, sym_full, atoms;
                        Ecut=Ecut, tol=tol)
        basis_full, Psi_full, orben_full, ρ_full = res
        test_orthonormality(basis_full, Psi_full, tol=tol)

        kcoords, ksymops = bzmesh_ir_wedge(kgrid_size, testcase.lattice, atoms)
        res = get_bands(testcase, kcoords, ksymops, atoms;
                        Ecut=Ecut, tol=tol)
        basis_ir, Psi_ir, orben_ir, ρ_ir = res

        test_orthonormality(basis_ir, Psi_ir, tol=tol)

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
                range = 1:length(orben_full[ikfull]) - n_ignore
                @test orben_full[ikfull][range] ≈ orben_ir[ik][range] atol=tol
            end
        end
    end

    test_full_vs_irreducible(silicon, [3, 3, 3], Ecut=5, tol=1e-6)
    test_full_vs_irreducible(silicon, [2, 3, 4], Ecut=5, tol=1e-6)
    test_full_vs_irreducible(magnesium, [2, 3, 4], Ecut=5, tol=1e-6, n_ignore=1)
    test_full_vs_irreducible(aluminium, [1, 3, 5], Ecut=3, tol=1e-5, n_ignore=2)
    #
    # That's pretty expensive:
    # test_full_vs_irreducible([4, 4, 4], Ecut=5, tol=1e-6)
end
