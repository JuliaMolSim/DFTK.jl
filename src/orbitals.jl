# select the occupied orbitals assuming an insulator
function select_occupied_orbitals(basis::PlaneWaveBasis, ψ)
    model  = basis.model
    n_spin = model.n_spin_components
    @assert iszero(basis.model.temperature)
    n_bands = div(model.n_electrons, n_spin * filled_occupation(model), RoundUp)
    [ψk[:, 1:n_bands] for ψk in ψ]
end

# Packing routines used in direct_minimization and newton algorithms.
# They pack / unpack sets of ψ's (or compatible arrays, such as hamiltonian
# applies and gradients) to make them compatible to be used in algorithms
# from IterativeSolvers.
# Some care is needed here : some operators (for instance K in newton.jl)
# are real-linear but not complex-linear. To overcome this difficulty, instead of
# seeing them as operators from C^N to C^N, we see them as
# operators from R^2N to R^2N. In practice, this is done with the
# reinterpret function from julia.
# /!\ pack_ψ does not share memory while unpack_ψ does

reinterpret_real(x) = reinterpret(real(eltype(x)), x)
reinterpret_complex(x) = reinterpret(Complex{eltype(x)}, x)

function pack_ψ(ψ)
    # TODO as an optimization, do that lazily? See LazyArrays
    vcat([vec(ψk) for ψk in ψ]...)
end

function unpack_ψ(x, sizes_ψ)
    n_bands = sizes_ψ[1][2]
    lengths = prod.(sizes_ψ)
    ends = cumsum(lengths)
    # We unsafe_wrap the resulting array to avoid a complicated type for ψ.
    # The resulting array is valid as long as the original x is still in live memory.
    map(1:length(sizes_ψ)) do ik
        unsafe_wrap(Array{complex(eltype(x))},
                    pointer(@views x[ends[ik]-lengths[ik]+1:ends[ik]]),
                    sizes_ψ[ik])
    end
end
