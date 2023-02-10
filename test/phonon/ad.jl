using Test
using DFTK
using LinearAlgebra
using StaticArrays
using ForwardDiff
using FiniteDifferences

@testset "Phonons: AD" begin

include("../testcases.jl")

# TODO: To speed-up computations on the CI, be smarter later.
fast = true

function compute_dynmat_fd(basis::PlaneWaveBasis{T}; scf_kwargs...) where {T}
    # TODO: Cannot use symmetries at all, see https://github.com/JuliaMolSim/DFTK.jl/issues/817
    @assert isone(only(basis.model.symmetries))
    model = basis.model
    cell = (; model.lattice, model.atoms, model.positions)
    n_atoms = length(model.positions)
    n_dim = model.n_dim
    dynamical_matrix = zeros(ComplexF64, (n_dim, n_atoms, n_dim, n_atoms))
    for τ in 1:n_atoms
        for γ in 1:n_dim
            displacement = zero.(model.positions)
            displacement[τ] = StaticArrays.setindex(displacement[τ], one(T), γ)
            dynamical_matrix_τγ = -FiniteDifferences.central_fdm(5, 1)(zero(T)) do ε
                cell_disp = (; lattice=eltype(ε).(cell.lattice), cell.atoms,
                             positions=ε*displacement .+ cell.positions)
                model_disp = Model(convert(Model{eltype(ε)}, model); cell_disp...)
                basis_disp = PlaneWaveBasis(basis, model_disp)
                scfres = self_consistent_field(basis_disp; scf_kwargs...)
                forces = compute_forces(scfres)
                hcat(Array.(forces)...)
            end
            dynamical_matrix[:, :, γ, τ] = dynamical_matrix_τγ[1:n_dim, :]
        end
    end
    reshape(dynamical_matrix, n_dim*n_atoms, n_dim*n_atoms)
end

@testset "LDA supercell consistency" begin
    cell  = silicon
    Ecut  = 5
    kgrid = fast ? [2, 1, 1] : [2, 3, 2]
    scf_kwargs = (; callback=identity)

    model = model_LDA(cell.lattice, cell.atoms, cell.positions; symmetries=false)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    dynamical_matrix_ad = compute_dynmat_ad(basis; scf_kwargs...)
    dynamical_matrix_fd = compute_dynmat_fd(basis; scf_kwargs...)
    @test maximum(abs.(dynamical_matrix_ad - dynamical_matrix_fd)) < 1e-1

    @testset "Unfolding" begin
        # Unfold the cell in a random direction.
        if fast
            supercell_size = [2, 1, 1]
            kgrid_supercell = [1, 1, 1]
        else
            rand_dir = rand(1:3)
            supercell_size = ones(3)
            supercell_size[rand_dir] = kgrid[rand_dir]
            kgrid_supercell = copy(kgrid)
            kgrid_supercell[rand_dir] = 1
        end
        supercell = create_supercell(cell.lattice, cell.atoms, cell.positions, supercell_size)
        model_supercell = model_LDA(supercell.lattice, supercell.atoms, supercell.positions;
                                    symmetries=false)
        basis_supercell = PlaneWaveBasis(model_supercell; Ecut, kgrid=kgrid_supercell)

        dynamical_matrix_supercell_ad = compute_dynmat_ad(basis_supercell; scf_kwargs...)

        # We convert back the eigenvalues in Hartree to simplify check.
        eigs_uc = phonon_eigenvalues(basis, dynamical_matrix_ad) / DFTK.hartree_to_cm⁻¹
        eigs_sc = phonon_eigenvalues(basis_supercell,
                                     dynamical_matrix_supercell_ad) / DFTK.hartree_to_cm⁻¹

        tol = 1e-4
        # Three eigenvalues should be close to zero in each cases.
        @test count(abs.(eigs_uc) .< tol) == 3
        @test count(abs.(eigs_sc) .< tol) == 3
        # The three other in the unit cell should match three in the supercell.
        matching_eigs = count(sum(map(eigs_uc) do λ
                                      abs.( (λ .- eigs_sc) / λ) .< tol
                                  end) .> 0)
        @test matching_eigs ≥ 3
    end
end

end
