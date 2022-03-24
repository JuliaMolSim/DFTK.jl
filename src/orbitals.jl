# Returns the occupied orbitals, the occupation array and optionally the eigenvalues without
# virtual states (or states with small occupation level for metals).
# threshold is a parameter to distinguish between states we want to keep and the
# others when using temperature. It is set to 0.0 by default, to treat with insulators.
function select_occupied_orbitals(basis, ψ, occupation, eigenvalues=nothing; threshold=0.0)
    N = [findlast(x -> x > threshold, occk) for occk in occupation]
    selected_ψ   = [@view ψk[:, 1:N[ik]] for (ik, ψk)   in enumerate(ψ)]
    selected_occ = [      occk[1:N[ik]]  for (ik, occk) in enumerate(occupation)]

    # if we have an insulator, sanity check that the orbitals we kept are the
    # occupied ones
    if threshold == 0.0
        model   = basis.model
        n_spin  = model.n_spin_components
        n_bands = div(model.n_electrons, n_spin * filled_occupation(model), RoundUp)
        @assert n_bands == size(selected_ψ[1], 2)
    end

    if isnothing(eigenvalues)
        (ψ=selected_ψ, occupation=selected_occ)
    else
        selected_evals = [evalk[1:N[ik]] for (ik, evalk) in enumerate(eigenvalues)]
        (ψ=selected_ψ, occupation=selected_occ, eigenvalues=selected_evals)
    end
end

function select_occupied_orbitals(scfres::NamedTuple; threshold=0.0)
    truncated = select_occupied_orbitals(scfres.basis, scfres.ψ, scfres.occupation,
                                         scfres.eigenvalues; threshold)

    min_trunc_bands = minimum(length, scfres.occupation) - maximum(length, truncated.occupation)
    @assert min_trunc_bands ≥ scfres.n_ep_extra  # Ensure extra bands are truncated
    merge(scfres, truncated, (; n_ep_extra=0))
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

# Returns pointers into the unpacked ψ
# /!\ The resulting array is only valid as long as the original x is still in live memory.
function unsafe_unpack_ψ(x, sizes_ψ)
    lengths = prod.(sizes_ψ)
    ends = cumsum(lengths)
    # We unsafe_wrap the resulting array to avoid a complicated type for ψ.    
    map(1:length(sizes_ψ)) do ik
        unsafe_wrap(Array{complex(eltype(x))},
                    pointer(@views x[ends[ik]-lengths[ik]+1:ends[ik]]),
                    sizes_ψ[ik])
    end
end
unpack_ψ(x, sizes_ψ) = deepcopy(unsafe_unpack_ψ(x, sizes_ψ))

function random_orbitals(basis::PlaneWaveBasis{T}, kpt::Kpoint, howmany) where {T}
    ortho_qr(randn(Complex{T}, length(G_vectors(basis, kpt)), howmany))
end
