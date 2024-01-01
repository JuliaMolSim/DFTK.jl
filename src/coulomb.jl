function compute_poisson_green_coeffs(basis::PlaneWaveBasis{T};
                                      scaling_factor=one(T),
                                      q=zero(Vec3{T})) where {T}
    model = basis.model

    # Solving the Poisson equation ΔV = -4π ρ in Fourier space
    # is multiplying elementwise by 4π / |G|^2.
    poisson_green_coeffs = 4T(π) ./ [sum(abs2, model.recip_lattice * (G + q))
                                     for G in to_cpu(G_vectors(basis))]
    if iszero(q)
        # Compensating charge background => Zero DC.
        GPUArraysCore.@allowscalar poisson_green_coeffs[1] = 0
        # Symmetrize Fourier coeffs to have real iFFT.
        enforce_real!(poisson_green_coeffs, basis)
    end
    poisson_green_coeffs = to_device(basis.architecture, poisson_green_coeffs)
    scaling_factor .* poisson_green_coeffs
end


raw"""
Compute the Coulomb vertex
```math
Γ_{km,\tilde{k}n,G} = ∫_Ω \sqrt{\frac{4π}{|G|^2} e^{-ir\cdot G} ψ_{km}^∗ ψ_{\tilde{k}n} dr
```
where `n_bands` is the number of bands to be considered.
"""
@timing function compute_coulomb_vertex(basis,
                                        ψ::AbstractVector{<:AbstractArray{T}};
                                        n_bands=size(ψ[1], 2)) where {T}
    mpi_nprocs(basis.comm_kpts) > 1 && error("Cannot use mpi")
    if length(basis.kpoints) > 1 && basis.use_symmetries_for_kpoint_reduction
        error("Cannot use symmetries right now.")
        # This requires appropriate insertion of kweights
    end

    n_G   = prod(basis.fft_size)
    n_kpt = length(basis.kpoints)
    ΓmnG  = zeros(complex(T), n_kpt, n_bands, n_kpt, n_bands, n_G)
    @views for (ikn, kptn) in enumerate(basis.kpoints), n = 1:n_bands
        ψnk_real = ifft(basis, kptn, ψ[ikn][:, n])
        for (ikm, kptm) in enumerate(basis.kpoints)
            q = kptn.coordinate - kptm.coordinate
            coeffs = sqrt.(compute_poisson_green_coeffs(basis; q))
            for m in 1:n_bands
                ψmk_real = ifft(basis, kptm, ψ[ikm][:, m])
                ΓmnG[ikm, m, ikn, n, :] = coeffs .* fft(basis, conj(ψmk_real) .* ψnk_real)
            end  # ψmk
        end # kptm
    end  # kptn, ψnk
    ΓmnG
end
function compute_coulomb_vertex(scfres::NamedTuple)
    compute_coulomb_vertex(scfres.basis, scfres.ψ; n_bands=scfres.n_bands_converge)
end

function svdcompress_coulomb_vertex(ΓmnG::AbstractArray{T,5}; tol=1e-10) where {T}
    Γmat = reshape(ΓmnG, prod(size(ΓmnG)[1:4]), size(ΓmnG, 5))

    F = svd(Γmat)
    nkeep = findlast(s -> abs(s) > tol, F.S)
    @views if isnothing(nkeep)
        ΓmnG
    else
        cΓmat = F.U[:, 1:nkeep] * Diagonal(F.S[1:nkeep])
        reshape(cΓmat, size(ΓmnG)[1:4]..., nkeep)
    end
end
