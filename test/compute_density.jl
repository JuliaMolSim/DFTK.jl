using Test
using LinearAlgebra
using DFTK
include("testcases.jl")

# TODO Once we have converged SCF densities in a file it would be better to instead / also
#      test the energies of these densities and compare them directly to the reference
#      energies obtained in the data files


@testset "Using BZ symmetry yields identical density" begin
    function get_bands(testcase, fft_size, kcoords, ksymops, composition...;
                       Ecut=5, tol=1e-8)
        model = model_dft(testcase.lattice, :lda_xc_teter93, composition...)
        basis = PlaneWaveModel(model, fft_size, Ecut, kcoords, ksymops)
        ham = Hamiltonian(basis, guess_gaussian_sad(basis, composition...))

        n_bands = 4
        res = lobpcg(ham, n_bands; prec=PreconditionerKinetic(ham, α=0.1), tol=tol)

        @assert testcase.n_electrons == 8
        occ = DFTK.find_occupation_around_fermi(basis, res.λ, res.X)
        ρnew = compute_density(basis, res.X, occ)

        basis, res.X, res.λ, ρnew
    end

    function test_orthonormality(basis, Psi; tol=1e-8)
        n_k = length(Psi)
        n_states = size(Psi[1], 2)
        n_fft = prod(basis.fft_size)

        for (ik, kpt) in enumerate(basis.kpoints)
            # Fourier-transform the wave functions to real space
            Ψk = Psi[ik]
            Ψk_real = DFTK.G_to_r(basis, kpt, Ψk)

            # TODO I am not quite sure why this is needed here maybe this points at an
            #      error in the normalisation of the Fourier transform.
            #      This is also done in the compute_density routine inside the
            #      core/compute_density.jl file
            Ψk_real /= sqrt(basis.model.unit_cell_volume)

            T = real(eltype(Ψk_real))
            Ψk_real_mat = reshape(Ψk_real, n_fft, n_states)
            Ψk_real_overlap = adjoint(Ψk_real_mat) * Ψk_real_mat
            nondiag = Ψk_real_overlap - I * (n_fft / basis.model.unit_cell_volume)

            @test maximum(abs.(nondiag)) < tol
        end
    end

    function test_full_vs_irreducible(kgrid_size; Ecut=5, tol=1e-8)
        testcase = silicon
        Si = Species(testcase.atnum, psp=load_psp(testcase.psp))
        composition = (Si => testcase.positions, )

        kfull, sym_full = bzmesh_uniform(kgrid_size)
        fft_size = determine_grid_size(testcase.lattice, Ecut)
        res = get_bands(testcase, fft_size, kfull, sym_full,
                        composition...; Ecut=Ecut, tol=tol)
        basis_full, Psi_full, orben_full, ρ_full = res
        test_orthonormality(basis_full, Psi_full, tol=tol)

        kcoords, ksymops = bzmesh_ir_wedge(kgrid_size, testcase.lattice, composition...)
        basis_ir, Psi_ir, orben_ir, ρ_ir = get_bands(testcase, fft_size, kcoords, ksymops,
                                                     composition...; Ecut=Ecut, tol=tol)
        test_orthonormality(basis_ir, Psi_ir, tol=tol)

        # Test density is equivalent
        @test maximum(abs.(ρ_ir - ρ_full)) < 10tol

        # Test equivalent k-Points have the same orbital energies
        for (ik, k) in enumerate(kcoords)
            for (S, τ) in ksymops[ik]
                ikfull = findfirst(1:length(kfull)) do idx
                    all(elem.den == 1 for elem in (S * k - kfull[idx]))
                end
                @test ikfull !== nothing

                # Orbital energies should agree (by symmetry)
                @test orben_full[ikfull] ≈ orben_ir[ik] atol=tol
            end
        end
    end

    test_full_vs_irreducible([3, 3, 3], Ecut=5, tol=1e-6)
    test_full_vs_irreducible([2, 3, 4], Ecut=5, tol=1e-6)
    #
    # That's pretty expensive:
    # test_full_vs_irreducible([4, 4, 4], Ecut=5, tol=1e-6)
end
