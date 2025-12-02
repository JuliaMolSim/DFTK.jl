struct EstimatedMemoryUsage
    n_kpoints::Int
    n_pw::Int
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
    n_kpoints = mpi_max(length(basis.kpoints), basis.comm_kpts)
    n_pw = mpi_max(length(basis.kpoints[1].G_vectors), basis.comm_kpts)
    n_bands = AdaptiveBands(model).n_bands_compute

    ψk_bytes = sizeof(Complex{T}) * n_pw * n_bands
    ψ_bytes = ψk_bytes * n_kpoints

    n_fftpoints = prod(basis.fft_size)
    ρ_bytes = sizeof(T) * n_fftpoints

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

    nonlocal_Pk_bytes = sizeof(Complex{T}) * n_pw * n_nonlocal_projectors
    nonlocal_P_bytes = nonlocal_Pk_bytes * n_kpoints

    # For the SCF, we estimate the following:
    # - 1x the nonlocal projectors
    # - 2x the full ψ (current + new) and 6 extra ψk for the LOBPCG
    # - 12x the density (current + new + default Anderson history of 10)
    scf_peak_bytes = (nonlocal_P_bytes
                      + 2 * ψ_bytes + 6 * ψk_bytes
                      + 12 * ρ_bytes)

    EstimatedMemoryUsage(
        n_kpoints, n_pw, n_bands, n_nonlocal_projectors,
        ψk_bytes, nonlocal_Pk_bytes,
        ψ_bytes, ρ_bytes, nonlocal_P_bytes, scf_peak_bytes)
end