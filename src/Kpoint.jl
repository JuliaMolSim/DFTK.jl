"""
Discretization information for ``k``-point-dependent quantities such as orbitals.
More generally, a ``k``-point is a block of the Hamiltonian;
e.g. collinear spin is treated by doubling the number of ``k``-points.
"""
struct Kpoint{T <: Real, GT <: AbstractVector{Vec3{Int}}}
    spin::Int                        # Spin component can be 1 or 2 as index into what is
    #                                # returned by the `spin_components` function
    coordinate::Vec3{T}              # Fractional coordinate of k-point
    G_vectors::GT                    # Wave vectors in integer coordinates (vector of Vec3{Int})
    #                                # ({G, 1/2 |k+G|^2 ≤ Ecut})
                                     # This is not assumed to be in any particular order
    mapping::Vector{Int}             # Index of G_vectors[i] on the FFT grid:
    #                                # G_vectors(basis)[kpt.mapping[i]] == G_vectors(basis, kpt)[i]
    mapping_inv::Dict{Int, Int}      # Inverse of `mapping`:
    #                                # G_vectors(basis)[i] == G_vectors(basis, kpt)[mapping_inv[i]]
    mapping_gpu::AbstractVector{Int} # GPU copy of `mapping` for fast FFTs
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
    mapping_gpu = to_device(architecture, mapping)
    Kpoint(spin, k, Gvecs_k, mapping, mapping_inv, mapping_gpu)
end

# Construct the kpoint with coordinate equivalent_kpt.coordinate + ΔG.
# Equivalent to (but faster than) Kpoint(equivalent_kpt.coordinate + ΔG).
function construct_from_equivalent_kpt(basis, equivalent_kpt, coordinate, ΔG)
    linear = LinearIndices(basis.fft_size)
    # Mapping is the same as if created from scratch, although it is not ordered.
    mapping = map(CartesianIndices(basis.fft_size)[equivalent_kpt.mapping]) do G
        linear[CartesianIndex(mod1.(Tuple(G + CartesianIndex(ΔG...)), basis.fft_size))]
    end
    mapping_inv = Dict(ifull => iball for (iball, ifull) in enumerate(mapping))
    mapping_gpu = to_device(basis.architecture, mapping)
    Kpoint(equivalent_kpt.spin, Vec3(coordinate), equivalent_kpt.G_vectors .+ Ref(ΔG),
           mapping, mapping_inv, mapping_gpu)
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
                                      kpt.G_vectors, kpt.mapping,
                                      kpt.mapping_inv, kpt.mapping_gpu))
        end
    end
    all_kpoints
end

# Forward FFT calls taking a Kpoint as argument
ifft!(f_real::AbstractArray3, fft_grid::FFTGrid, kpt::Kpoint,
      f_fourier::AbstractVector; normalize=true) =
    ifft!(f_real, fft_grid, kpt.mapping_gpu, f_fourier; normalize=normalize)

ifft(fft_grid::FFTGrid, kpt::Kpoint, f_fourier::AbstractVector; kwargs...) =
    ifft(fft_grid, kpt.mapping_gpu, f_fourier; kwargs...)

fft!(f_fourier::AbstractVector, fft_grid::FFTGrid, kpt::Kpoint,
     f_real::AbstractArray3; normalize=true) =
     fft!(f_fourier, fft_grid, kpt.mapping_gpu, f_real; normalize=normalize)

fft(fft_grid::FFTGrid, kpt::Kpoint, f_real::AbstractArray3; kwargs...) =
    fft(fft_grid, kpt.mapping_gpu, f_real; kwargs...)