"""
    G_vectors(fft_size::Tuple)


of given sizes.
"""
function G_vectors(fft_size::Union{Tuple,AbstractVector})
    # Note that a collect(G_vectors_generator(fft_size)) is 100-fold slower
    # than this implementation, hence the code duplication.
    start = .- cld.(fft_size .- 1, 2)
    stop  = fld.(fft_size .- 1, 2)
    axes  = [[collect(0:stop[i]); collect(start[i]:-1)] for i = 1:3]
    [Vec3{Int}(i, j, k) for i in axes[1], j in axes[2], k in axes[3]]
end

function G_vectors_generator(fft_size::Union{Tuple,AbstractVector})
    # The generator version is used mainly in symmetry.jl for lowpass_for_symmetry! and
    # accumulate_over_symmetries!, which are 100-fold slower with G_vector(fft_size).
    start = .- cld.(fft_size .- 1, 2)
    stop  = fld.(fft_size .- 1, 2)
    axes = [[collect(0:stop[i]); collect(start[i]:-1)] for i = 1:3]
    (Vec3{Int}(i, j, k) for i in axes[1], j in axes[2], k in axes[3])
end

"""
Discretization information for ``k``-point-dependent quantities such as orbitals.
More generally, a ``k``-point is a block of the Hamiltonian;
e.g. collinear spin is treated by doubling the number of ``k``-points.
"""
struct Kpoint{T <: Real, GT <: AbstractVector{Vec3{Int}}}
    spin::Int                     # Spin component can be 1 or 2 as index into what is
    #                             # returned by the `spin_components` function
    coordinate::Vec3{T}           # Fractional coordinate of k-point
    G_vectors::GT                 # Wave vectors in integer coordinates (vector of Vec3{Int})
    #                             # ({G, 1/2 |k+G|^2 ≤ Ecut})
                                  # This is not assumed to be in any particular order
    mapping::Vector{Int}          # Index of G_vectors[i] on the FFT grid:
    #                             # G_vectors(basis)[kpt.mapping[i]] == G_vectors(basis, kpt)[i]
    mapping_inv::Dict{Int, Int}   # Inverse of `mapping`:
    #                             # G_vectors(basis)[i] == G_vectors(basis, kpt)[mapping_inv[i]]
end

function Kpoint(spin::Integer, coordinate::AbstractVector{<:Real},
                recip_lattice::AbstractMatrix{T}, fft_size, Ecut;
                variational=true, architecture::AbstractArchitecture) where {T}
    mapping = Int[]
    Gvecs_k = Vec3{Int}[]
    k = Vec3{T}(coordinate)
    # provide a rough hint so that the arrays don't have to be resized so much
    n_guess = div(prod(fft_size), 8)
    sizehint!(mapping, n_guess)
    sizehint!(Gvecs_k, n_guess)
    for (i, G) in enumerate(G_vectors(fft_size))
        if !variational || norm2(recip_lattice * (G + k)) / 2 ≤ Ecut
            push!(mapping, i)
            push!(Gvecs_k, G)
        end
    end
    Gvecs_k = to_device(architecture, Gvecs_k)

    mapping_inv = Dict(ifull => iball for (iball, ifull) in enumerate(mapping))
    Kpoint(spin, k, Gvecs_k, mapping, mapping_inv)
end

# Construct the kpoint with coordinate equivalent_kpt.coordinate + ΔG.
# Equivalent to (but faster than) Kpoint(equivalent_kpt.coordinate + ΔG).
function construct_from_equivalent_kpt(fft_size, equivalent_kpt, coordinate, ΔG)
    linear = LinearIndices(fft_size)
    # Mapping is the same as if created from scratch, although it is not ordered.
    mapping = map(CartesianIndices(fft_size)[equivalent_kpt.mapping]) do G
        linear[CartesianIndex(mod1.(Tuple(G + CartesianIndex(ΔG...)), fft_size))]
    end
    mapping_inv = Dict(ifull => iball for (iball, ifull) in enumerate(mapping))
    Kpoint(equivalent_kpt.spin, Vec3(coordinate), equivalent_kpt.G_vectors .+ Ref(ΔG),
           mapping, mapping_inv)
end

@timing function build_kpoints(model::Model{T}, fft_size, kcoords, Ecut;
                               variational=true,
                               architecture::AbstractArchitecture) where {T}
    # Build all k-points for the first spin
    kpoints_spin_1 = [Kpoint(1, k, model.recip_lattice, fft_size, Ecut;
                             variational, architecture)
                      for k in kcoords]
    all_kpoints = similar(kpoints_spin_1, 0)
    for iσ = 1:model.n_spin_components
        for kpt in kpoints_spin_1
            push!(all_kpoints, Kpoint(iσ, kpt.coordinate,
                                      kpt.G_vectors, kpt.mapping, kpt.mapping_inv))
        end
    end
    all_kpoints
end

