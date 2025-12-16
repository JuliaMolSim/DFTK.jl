using ProgressMeter

# similar to compute_poisson_green_coeffs from hartree.jl but
# returns coulomb_potential only on spherical grid (G-q)² < 2Ecut
function compute_coulomb_kernel(basis::PlaneWaveBasis{T};
                                scaling_factor=one(T),
                                q=zero(Vec3{T})) where {T}
    model = basis.model
    kpoint = basis.kpoints[1] # only works for Gamma-only (need qpoint otherwise)

    coulomb_potential = 4T(π) ./ [sum(abs2, model.recip_lattice * (G - q))  # is it G-q or G+q? TODO
                                     for G in to_cpu(kpoint.G_vectors)]
    
    if iszero(q)
        # Compensating charge background => Zero DC.
        GPUArraysCore.@allowscalar coulomb_potential[1] = 0
        
        # do we really need that?
        ## Symmetrize Fourier coeffs to have real iFFT.
        #enforce_real!(coulomb_potential, basis)
    end
    coulomb_potential = to_device(basis.architecture, coulomb_potential)
    scaling_factor .* coulomb_potential
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

    # show progress via ProgressMeter
    progress = Progress(n_bands*size(basis.kpoints,1); desc="Compute Coulomb vertices", dt=0.5, barlen=20, color=:black)
    update!(progress, 0)
    flush(stdout)

    kpt   = basis.kpoints[1]
    n_G   = length(kpt.G_vectors) # works only for 1-kpoint
    n_kpt = length(basis.kpoints)
    ΓmnG  = zeros(complex(T), n_kpt, n_bands, n_kpt, n_bands, n_G)
    @views for (ikn, kptn) in enumerate(basis.kpoints), n = 1:n_bands
        ψnk_real = ifft(basis, kptn, ψ[ikn][:, n])
        for (ikm, kptm) in enumerate(basis.kpoints)
            q = kptn.coordinate - kptm.coordinate
            coeffs = sqrt.(compute_coulomb_kernel(basis; q))
            for m in 1:n_bands
                ψmk_real = ifft(basis, kptm, ψ[ikm][:, m])
                ΓmnG[ikm, m, ikn, n, :] = coeffs .* fft(basis, kptn, conj(ψmk_real) .* ψnk_real) # kptn has to be some qptn (but works for Gamma-only)
            end  # ψmk
        end # kptm
        next!(progress)
    end  # kptn, ψnk
    ΓmnG
end
function compute_coulomb_vertex(scfres::NamedTuple)
    compute_coulomb_vertex(scfres.basis, scfres.ψ; n_bands=scfres.n_bands_converge)
end


# CoulombGramian E(G,G') defined as E = -Γ^† * Γ, where Γ(ab,G) = <a|G|b>
# see Eq. (10) in Hummel et al., JCTC (doi.org/10.1063/1.4977994).
# The operator CoulombGramian enables efficient application to a vector
# E*v = -Γ^† * (Γ*v) without full construction of E(G,G')
# in order to diagonalize E through iterative methods.
struct CoulombGramian{T}
    Γmat::T
end
function LinearAlgebra.mul!(Y, op::CoulombGramian, X)
    T = eltype(op)
    Ywork = zeros(T, size(op.Γmat,1), size(X,2))
    mul!(Ywork, op.Γmat, X, -1.0, 0.0)
    mul!(Y, op.Γmat', Ywork)
    return Y
end
function Base.:*(op::CoulombGramian, X::AbstractMatrix)
    T_out = promote_type(eltype(op), eltype(X))
    Y = similar(X, T_out)
    mul!(Y, op, X) 
    return Y
end
Base.size(op::CoulombGramian) = (size(op.Γmat, 2), size(op.Γmat, 2))
Base.eltype(op::CoulombGramian) = eltype(op.Γmat)
LinearAlgebra.ishermitian(op::CoulombGramian) = true

# thresh is in units of energy (Hartree)
function svdcompress_coulomb_vertex(ΓmnG::AbstractArray{T,5}; thresh=1e-6) where {T}
    Γmat = reshape(ΓmnG, prod(size(ΓmnG)[1:4]), size(ΓmnG, 5))

    NFguess = round(Int, 10*size(Γmat,1)^0.5)
    NG = size(Γmat,2)
    ϕk = randn(ComplexF64, NG, NFguess)
    for a in 1:NFguess
        ϕk[:,a] ./= norm(ϕk[:,a]) # normalize
    end
    
    E_GG = CoulombGramian(Γmat)

    # estimate the required time (assuming init + 1 iteration)
    flop_count = 2 * (2 * prod(size(Γmat)) * NFguess)     # 2 x application of E_GG
    flop_count *= 2                                       # init + first iteration
    flop_count += 2 * size(Γmat,2) * (3*NFguess)^2        # orthogonalization
    flop_count *= (eltype(E_GG) <: Complex) ? 4 : 1       # times 4 for complex cases
    flop_rate = 0.8*LinearAlgebra.peakflops(500) # assume 80% of peakflops
    estimated_seconds = flop_count / flop_rate
    time_str = if estimated_seconds < 10
        "a few seconds"
    elseif estimated_seconds < 120
        "$(round(Int, estimated_seconds)) seconds"
    elseif estimated_seconds < 7200
        "$(round(Int, estimated_seconds / 60)) minutes"
    else
        "$(round(estimated_seconds / 3600, digits=1)) hours"
    end
    println("Compress Coulomb vertices. Estimated time (at $(round(Int, flop_rate/1e9)) GFLOPS): $time_str")

    lobpcg_thresh = thresh/10 # thresh for LOBPCG should be smaller than thresh
    @time FFF = lobpcg_hyper(E_GG, ϕk; tol=lobpcg_thresh) 
    nkeep = findlast(s -> abs(s) > thresh, FFF.λ)
    println("Compressed Coulomb vertices from NG=$(NG) to NF=$(nkeep).")
    @views if isnothing(nkeep)
        ΓmnG
    else
        cΓmat = Γmat * FFF.X[:, 1:nkeep] 
        reshape(cΓmat, size(ΓmnG)[1:4]..., nkeep)
    end
     
    #@time U, S, V = tsvd(Γmat, NFguess; tolconv=tolconv, maxiter=maxiter)
    ##@time res = eigen(Γmat' * Γmat)
    #@time F = svd(Γmat)

    #Serror = abs.(S - F.S[1:NFguess])
    ##println(Serror)
    ##println(" ")
    #println("max error at: ", findmax(Serror))

    #tol = sqrt(thresh) # singular values are sqrt of energies
    #nkeep = findlast(s -> abs(s) > tol, F.S)
    #@views if isnothing(nkeep)
    #    ΓmnG
    #else
    #    cΓmat = F.U[:, 1:nkeep] * Diagonal(F.S[1:nkeep])
    #    reshape(cΓmat, size(ΓmnG)[1:4]..., nkeep)
    #end
end
