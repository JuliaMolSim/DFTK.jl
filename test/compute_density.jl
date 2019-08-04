using Test
using LinearAlgebra
using DFTK
using DFTK: empty_potential, update_energies_potential!, G_to_r!
include("silicon_testcases.jl")

# TODO Once we have converged SCF densities in a file it would be better to instead / also
#      test the energies of these densities and compare them directly to the reference
#      energies obtained in the data files


@testset "Using BZ symmetry yields identical density" begin
    function get_bands(lattice, Ggrid, kpoints, ksymops, composition...;
                       Ecut=5, tol=1e-8)
        kweights = [length(symops) for symops in ksymops]
        kweights = kweights / sum(kweights)
        basis = PlaneWaveBasis(lattice, Ggrid, Ecut, kpoints, kweights, ksymops)

        ham = Hamiltonian(
            basis,
            pot_local=build_local_potential(basis, composition...),
            pot_nonlocal=build_nonlocal_projectors(basis, composition...),
            pot_hartree=PotHartree(basis),
            pot_xc=PotXc(basis, :lda_xc_teter93)
        )
        ρ = guess_gaussian_sad(basis, composition...)
        values_hartree = empty_potential(ham.pot_hartree)
        values_xc = empty_potential(ham.pot_xc)
        energies = Dict{Symbol, real(eltype(ρ))}()
        update_energies_potential!(energies, values_hartree, ham.pot_hartree, ρ)
        update_energies_potential!(energies, values_xc, ham.pot_xc, ρ)

        res = lobpcg(ham, 4; pot_hartree_values=values_hartree,
                     pot_xc_values=values_xc, prec=PreconditionerKinetic(ham, α=0.1),
                     tol=tol)
        occ = [2ones(4) for _ in 1:length(kpoints)]  # TODO This Si specific
        ρnew = compute_density(basis, res.X, occ)

        basis, res.X, res.λ, ρnew
    end

    function test_orthonormality(basis, Psi; tol=1e-8)
        n_k = length(Psi)
        n_states = size(Psi[1], 2)
        n_fft = prod(size(basis.FFT))

        for ik in 1:n_k
            Ψk = Psi[ik]

            # Fourier-transform the wave functions to real space
            Ψk_real = similar(Ψk[:, 1], size(basis.FFT)..., n_states)
            G_to_r!(basis, Ψk, Ψk_real, gcoords=basis.basis_wf[ik])

            # TODO I am not quite sure why this is needed here maybe this points at an
            #      error in the normalisation of the Fourier transform.
            #      This is also done in the compute_density routine inside the
            #      core/compute_density.jl file
            Ψk_real /= sqrt(basis.unit_cell_volume)

            T = real(eltype(Ψk_real))
            Ψk_real_mat = reshape(Ψk_real, n_fft, n_states)
            Ψk_real_overlap = adjoint(Ψk_real_mat) * Ψk_real_mat
            nondiag = Ψk_real_overlap - I * (n_fft / basis.unit_cell_volume)

            @test maximum(abs.(nondiag)) < tol
        end
    end

    function test_full_vs_irreducible(kgrid_size; Ecut=5, tol=1e-8)
        Si = Species(atnum, psp=load_psp("si-pade-q4.hgh"))

        kfull, sym_full = bzmesh_uniform(kgrid_size)
        Ggrid = determine_grid_size(lattice, Ecut; kpoints=kfull) * ones(3)
        res = get_bands(lattice, Ggrid, kfull, sym_full,
                        Si => positions, Ecut=Ecut, tol=tol)
        basis_full, Psi_full, orben_full, ρ_full = res
        test_orthonormality(basis_full, Psi_full, tol=tol)

        kpoints, ksymops = bzmesh_ir_wedge(kgrid_size, lattice, Si => positions)
        basis_ir, Psi_ir, orben_ir, ρ_ir = get_bands(lattice, Ggrid, kpoints, ksymops,
                                                     Si => positions, Ecut=Ecut, tol=tol)
        test_orthonormality(basis_ir, Psi_ir, tol=tol)

        # Test density is equivalent
        @test maximum(abs.(ρ_ir - ρ_full)) < 10tol

        # Test equivalent k-Points have the same orbital energies
        for (ik, k) in enumerate(kpoints)
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
