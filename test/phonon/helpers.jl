using Random
using LinearAlgebra

# Helpers functions for tests.
# TODO: Temporary, explanations too scarce. To be changed with proper phonon computations.

function ph_compute_reference(payload, model_supercell)
    n_atoms = length(model_supercell.positions)
    n_dim = model_supercell.n_dim
    T = eltype(model_supercell.lattice)
    dynmat_ad = zeros(T, 3, n_atoms, 3, n_atoms)
    term = only(model_supercell.term_types)
    for τ in 1:n_atoms
        for γ in 1:n_dim
            displacement = zero.(model_supercell.positions)
            displacement[τ] = setindex(displacement[τ], one(T), γ)
            dynmat_ad[:, :, γ, τ] = -ForwardDiff.derivative(zero(T)) do ε
                lattice = convert(Matrix{eltype(ε)}, model_supercell.lattice)
                positions = ε*displacement .+ model_supercell.positions
                (; forces) = payload(term, lattice, model_supercell.atoms, positions)
                hcat(Array.(forces)...)
            end
        end
    end
    hessian_ad = DFTK.dynmat_red_to_cart(model_supercell, dynmat_ad)
    sort(compute_squared_frequencies(hessian_ad))
end



# We do not take the square root to compare results with machine precision.
function compute_squared_frequencies(matrix)
    n, m = size(matrix, 1), size(matrix, 2)
    Ω = eigvals(reshape(matrix, n*m, n*m))
    real(Ω)
end

function generate_random_supercell(; max_length=6)
    n_max = min(max_length, 5)
    supercell_size = nothing
    while true
        supercell_size = rand(1:n_max, 3)
        prod(supercell_size) < max_length && break
    end
    supercell_size
end

function generate_supercell_qpoints(; supercell_size=generate_random_supercell())
    qpoints_list = Iterators.product([1:n_sc for n_sc in supercell_size]...)
    qpoints = map(qpoints_list) do n_sc
        DFTK.normalize_kpoint_coordinate.([n_sc[i] / supercell_size[i] for i in 1:3])
    end |> vec

    (; supercell_size, qpoints)
end
