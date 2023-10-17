using Random
using LinearAlgebra

# Helpers functions for tests.
# TODO: Temporary, explanations too scarce. To be changed with proper phonon computations.

# We do not take the square root to compare results with machine precision.
function compute_ω²(matrix)
    n, m = size(matrix)[1:2]
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
