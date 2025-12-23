using ProgressMeter

raw"""
    CoulombModel

Abstract type for different methods of computing the 
discretised Coulomb kernel ``v(G+q) = 4π/|G+q|²``.

Available models:
- [`ProbeCharge`](@ref): Gygi-Baldereschi probe charge method (default)
- [`NeglectSingularity`](@ref): Set G+q=0 component to zero
- [`SphericallyTruncated`](@ref): Spherical truncation at radius Rcut
- [`WignerSeitzTruncated`](@ref): Wigner-Seitz cell truncation

See also: [`compute_coulomb_kernel`](@ref)
"""
abstract type CoulombModel end


raw"""
    compute_coulomb_kernel(basis; scaling_factor=1, q=0, coulomb_model=ProbeCharge())

Compute Coulomb kernel, i.e. essentially ``v(G+q) = 4π/|G+q|²``, on spherical plane-wave grid.

Returns the Fourier-space Coulomb interaction for momentum transfer `q`,
evaluated only on the spherical cutoff |G+q|² < 2Ecut (not the full cubic FFT grid).

!!! note "Gamma-point only"
    Currently only works for single k-point calculations (Gamma-only).
    For general k-points, a q-dependent basis would be needed.

# Arguments
- `basis::PlaneWaveBasis`: Plane-wave basis defining the grid
- `scaling_factor=1`: Global scaling factor applied to the result
- `q=zero(Vec3)`: Momentum transfer vector in fractional coordinates
- `coulomb_model::CoulombModel=ProbeCharge()`: Method for treating singularity

# Returns
Vector of Coulomb kernel values for each G-vector in the spherical cutoff.
"""
function compute_coulomb_kernel(basis::PlaneWaveBasis{T};
                                scaling_factor=one(T),
                                q=zero(Vec3{T}),
                                coulomb_model::CoulombModel=ProbeCharge()) where {T}
    # currently only works for Gamma-only (need correct q-point otherwise)
    qpt = basis.kpoints[1] 
    coulomb_kernel =  _compute_coulomb_kernel(basis, qpt, q, coulomb_model)

    if iszero(q) # Symmetrize Fourier coeffs to have real iFFT.
        #enforce_real!(coulomb_kernel, basis) # enforce_real doesn't work as we don't use the full cubic G-grid
    end

    coulomb_kernel = to_device(basis.architecture, coulomb_kernel)
    scaling_factor .* coulomb_kernel
end


"""
    ProbeCharge <: CoulombModel

Probe charge Ewald method for treating the Coulomb singularity.

Uses a Gaussian probe charge with width parameter α = π²/Ecut to regularize
the G+q=0 component. Well-tested in VASP for production calculations.

# Reference
Phys. Rev. B 48, 5058 (1993), doi.org/10.1103/PhysRevB.48.5058
"""
struct ProbeCharge <: CoulombModel end
function _compute_coulomb_kernel(basis::PlaneWaveBasis{T},
                                 qpt::Kpoint,
                                 q::Vec3{T},
                                 coulomb_model::ProbeCharge) where {T}
    model = basis.model
    NG = length(qpt.G_vectors)
    coulomb_kernel = zeros(T, NG)
    # default Coulomb potential using probe charge method for singularity
    α = π^2 / basis.Ecut # well tested in VASP such that e^(-α*G²) is localized charge with full support on G grid
    probe_charge_sum = zero(T)
    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)
        Gnorm2 = sum(abs2, G_cart)
        found_singularity = (iG==1 && iszero(q))
        if !found_singularity
            coulomb_kernel[iG] = 4T(π) / Gnorm2
            probe_charge_sum += coulomb_kernel[iG] * exp(-α*Gnorm2)
        end
        if iG == NG # in the last cycle probe_charge_sum is complete
            Ω = model.unit_cell_volume  # volume of cell 
            probe_charge_integral = 8*π^2*sqrt(π/α) * Ω/(2π)^3  #  = Ω/(2π)^3 ∫ 4π/q² ρ(q) dq  with  ρ(q)=e^(-αq²)
            coulomb_kernel[1] = probe_charge_integral - probe_charge_sum
        end
    end
    return coulomb_kernel
end

"""
    NeglectSingularity <: CoulombModel

Simply set the G+q=0 Coulomb kernel component to zero.

This is the simplest approach but leads to brutally slow convergence with system size.
Useful for testing or comparison purposes.
"""
struct NeglectSingularity <: CoulombModel end
function _compute_coulomb_kernel(basis::PlaneWaveBasis{T},
                                 qpt::Kpoint,
                                 q::Vec3{T},
                                 coulomb_model::NeglectSingularity) where {T}
    model = basis.model
    NG = length(qpt.G_vectors)
    coulomb_kernel = zeros(T, NG)
    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)
        Gnorm2 = sum(abs2, G_cart)
        found_singularity = (iG==1 && iszero(q))
        if !found_singularity
            coulomb_kernel[iG] = 4T(π) / Gnorm2
        end
    end
    return coulomb_kernel
end


"""
    SphericallyTruncated(; Rcut=-1.0) <: CoulombModel

Spherical truncation of Coulomb interaction at radius Rcut.

If Rcut < 0 (default), uses Rcut = ∛(3V/(4π)) => V = 4/3π*Rcut^3
where V is the BvK cell volume.

# Reference
Phys. Rev. B 77, 193110 (2008), doi.org/10.1103/PhysRevB.77.193110
"""
struct SphericallyTruncated <: CoulombModel 
    Rcut::Float64
end
SphericallyTruncated(; Rcut=-1.0) = SphericallyTruncated(Rcut)
function _compute_coulomb_kernel(basis::PlaneWaveBasis{T},
                                 qpt::Kpoint,
                                 q::Vec3{T},
                                 coulomb_model::SphericallyTruncated) where {T}
    model = basis.model
    NG = length(qpt.G_vectors)
    coulomb_kernel = zeros(T, NG)
    if coulomb_model.Rcut < 0.0 
        # use default value: V_BvK = 4/3πRcut³
        Rcut = cbrt(basis.model.unit_cell_volume*3/4/π) # TODO: multiply V with number of k-points
    else
        Rcut = coulomb_model.Rcut
    end
    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)
        Gnorm2 = sum(abs2, G_cart)
        found_singularity = (iG==1 && iszero(q))
        if !found_singularity
            coulomb_kernel[iG] = 4T(π) / Gnorm2 * (1-cos(Rcut*Gnorm2^0.5))
        else
            coulomb_kernel[iG] = 2T(π)*Rcut^2
        end
    end
    return coulomb_kernel
end

"""
    WignerSeitzTruncated <: CoulombModel

Truncate Coulomb interaction at the Wigner-Seitz cell boundary.

# Reference
Phys. Rev. B 87, 165122, 2013 (doi.org/10.1103/PhysRevB.87.165122)
"""
struct WignerSeitzTruncated <: CoulombModel end
function _compute_coulomb_kernel(basis::PlaneWaveBasis{T},
                                 qpt::Kpoint,
                                 q::Vec3{T},
                                 coulomb_model::WignerSeitzTruncated) where {T}
    model = basis.model
    NG = length(qpt.G_vectors)
    coulomb_kernel = zeros(T, NG)
    # calculate inradius R_in of Wigner-Seitz cell
    # R_in is given by the largest possible 
    # R_in = (sum_i n_i * a*i) / 2 
    # with integers n_i and |R_in| <= a_min where a_min is the length of the smallest lattice vector.
    # The inequality allows us to restrict n_i by exploiting Cauchy-Schwarz, leading to 
    # |n_i| <= a_min * |b_i| / 2π 
    # where b_i are the reciprocal lattice vectors.
    L_min = minimum(norm, eachcol(model.lattice))
    n_bounds = zeros(Int, 3)
    for i in 1:3
        b_vec = model.recip_lattice[:, i]
        b_len = norm(b_vec)
        N = (L_min * b_len) / (2π)
        n_bounds[i] = ceil(Int, N)
    end
    nx, ny, nz = n_bounds # in case of a cubic cell nx=ny=nz=1
    # construct R_in
    R_in = T(Inf)
    for ix in -nx:nx, iy in -ny:ny, iz in -nz:nz # loop through all necessary integers 
        ix == 0 && iy == 0 && iz == 0 && continue
        R = model.lattice * [ix, iy, iz]
        d = norm(R) / 2 # distance from origin to perpendicular bisector plane = |R|/2
        R_in = min(R_in, d)
    end
    
    # Nyquist frequency of FFT grid
    G_Nyquist = minimum(basis.fft_size[d] / 2 * norm(model.recip_lattice[:, d]) for d in 1:3) 

    ε_target = 1e-12
    ε_min = exp(-0.5*G_Nyquist*R_in)  # required: G_Nyquist > -2*log(ε)/R_in (Appendix A.1 in paper)
    ε = max(ε_target, ε_min)
    α = sqrt(-log(ε)) / R_in          # range separation parameter

    # FFT of long-range term erf(αr)/r restricted to Wigner-Seitz cell 
    r_vectors = DFTK.r_vectors(basis)
    V_lr_real = zeros(Complex{T}, basis.fft_size...)
    for idx in CartesianIndices(V_lr_real)
        r_frac = r_vectors[idx]
        r_centered = r_frac .- round.(r_frac) # MIC
        r_cart = model.lattice * r_centered
        d_min = norm(r_cart)
        for dx in -1:1, dy in -1:1, dz in -1:1 # Check neighbors for non-orthorhombic cells
            r_shifted = r_centered - T[dx, dy, dz]
            d = norm(model.lattice * r_shifted)
            d_min = min(d_min, d)
        end
        if d_min > sqrt(eps(T))
            V_lr_real[idx] = erf(α * d_min) / d_min
        else
            V_lr_real[idx] = 2*α / sqrt(T(π))
        end
        V_lr_real[idx] *= exp(-im * 2*T(π) * dot(q, r_frac)) # add phase e^{-2πiqr}
    end
    coulomb_kernel_lr = real.(fft(basis, qpt, V_lr_real))
    coulomb_kernel_lr .*= sqrt(model.unit_cell_volume)
    
    # analytic short-range term + long-range term 
    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)
        Gnorm2 = sum(abs2, G_cart)
        found_singularity = (iG==1 && iszero(q))
        Rcut = cbrt(basis.model.unit_cell_volume*3/4/π)
        if !found_singularity
            coulomb_kernel[iG] = 4T(π) / Gnorm2 * (1 - exp(-Gnorm2/(4α^2))) + coulomb_kernel_lr[iG]
        else
            coulomb_kernel[iG] = T(π)/α^2 + coulomb_kernel_lr[iG]
        end
    end
    return coulomb_kernel
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
