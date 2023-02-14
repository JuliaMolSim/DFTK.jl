"""
Compute the dynamical matrix at ``Γ``-point in reciprocal space of a system using automatic
differentiation. In this particular case, it is equal to the Hessian of the system.
"""
function compute_dynmat_ad(basis::PlaneWaveBasis{T}; scf_kwargs...) where {T}
    # TODO: Cannot use symmetries at all, see https://github.com/JuliaMolSim/DFTK.jl/issues/817
    @assert isone(only(basis.model.symmetries))

    model = basis.model
    cell = (; model.lattice, model.atoms, model.positions)
    n_atoms = length(model.positions)
    n_dim = model.n_dim

    dynamical_matrix = zeros(eltype(basis), (n_dim, n_atoms, n_dim, n_atoms))
    for τ in 1:n_atoms
        for γ in 1:n_dim
            displacement = zero.(model.positions)
            displacement[τ] = setindex(displacement[τ], one(T), γ)
            dynamical_matrix_τγ = -ForwardDiff.derivative(zero(T)) do ε
                cell_disp = (; cell.lattice, cell.atoms,
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
