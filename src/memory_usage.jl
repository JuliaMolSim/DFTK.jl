"""
Struct to hold **approximate** memory usage statistics for a standard calculation.

The memory usage is estimated per MPI process, based on the following quantities:
`n_kpoints`: maximum number of k-points across all MPI processes.
`n_Gk`: number of G-vectors for one of the k-points, might differ slightly between
        k-points and processes.
`n_bands`: estimated number of bands computed during the SCF.
`n_nonlocal_projectors`: number of non-local projectors in the system.
"""
struct MemoryStatistics
    n_kpoints::Int
    n_Gk::Int
    n_bands::Int
    n_nonlocal_projectors::Int
    ψk_bytes::Int
    nonlocal_Pk_bytes::Int
    ψ_bytes::Int
    ρ_bytes::Int
    nonlocal_P_bytes::Int
    scf_peak_bytes::Int
end

"""
    estimate_memory_usage(model::Model, basis_args...; basis_kwargs...)

Calculate a rough estimate of the leading contributions to the memory usage
for a standard DFT calculation, based on the model and the arguments that would
be passed to `PlaneWaveBasis`.

For example:
```jl
DFTK.estimate_memory_usage(model; kgrid=[4,4,4], Ecut=15)
```
"""
function estimate_memory_usage(model::Model, basis_args...; basis_kwargs...)
    # Instantiate a basis with no terms for speed and low memory usage
    # (We cannot have 0 terms, so we pass a dummy Kinetic term)
    model_noterms = Model(model; symmetries=model.symmetries, terms=[Kinetic()])
    basis = PlaneWaveBasis(model_noterms, basis_args...; basis_kwargs...)

    estimate_memory_usage(basis, model)
end

function estimate_memory_usage(basis::PlaneWaveBasis{T}, model=basis.model) where {T}
    isnothing(model.εF) || error("Cannot estimate memory usage for models with a fixed "
                                 * "Fermi level, since the number of bands is not known.")

    n_kpoints = maximum(krange -> sum(length, krange), basis.krange_allprocs)
    n_Gk = length(basis.kpoints[1].G_vectors)
    n_bands = AdaptiveBands(model).n_bands_compute

    ψk_bytes = sizeof(Complex{T}) * n_Gk * n_bands
    ψ_bytes = ψk_bytes * n_kpoints

    n_fftpoints = prod(basis.fft_size)
    ρ_bytes = sizeof(T) * n_fftpoints * model.n_spin_components

    if any(term_type -> term_type isa AtomicNonlocal, model.term_types)
        n_nonlocal_projectors = sum(model.atoms) do atom
            if atom isa ElementPsp
                count_n_proj(atom.psp)
            else
                0
            end
        end
    else
        n_nonlocal_projectors = 0
    end

    nonlocal_Pk_bytes = sizeof(Complex{T}) * n_Gk * n_nonlocal_projectors
    nonlocal_P_bytes = nonlocal_Pk_bytes * n_kpoints

    # For the SCF, we estimate the following:
    # - 1x the nonlocal projectors
    # - 2x the full ψ (current + new) and 6 extra ψk for the LOBPCG
    # - 12x the density (current + new + default Anderson history of 10)
    # This is a quick guess, the factors can likely be improved.
    scf_peak_bytes = (nonlocal_P_bytes
                      + 2 * ψ_bytes + 6 * ψk_bytes
                      + 12 * ρ_bytes)

    MemoryStatistics(
        n_kpoints, n_Gk, n_bands, n_nonlocal_projectors,
        ψk_bytes, nonlocal_Pk_bytes,
        ψ_bytes, ρ_bytes, nonlocal_P_bytes, scf_peak_bytes)
end