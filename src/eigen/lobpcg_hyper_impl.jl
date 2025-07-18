# Implementation of the LOBPCG algorithm, seeking optimal performance
# and stability (it should be very, very hard to break, even at tight
# tolerance criteria). The code is not 100% optimized yet, but the
# most time-consuming parts (BLAS3 and matvecs) should be OK. The
# implementation follows the scheme of Hetmaniuk & Lehoucq (see also
# the refinements in Duersch et. al.), with the following
# modifications:

# - Cholesky factorizations are used instead of the eigenvalue
# decomposition of Stathopolous & Wu for the orthogonalization.
# Cholesky orthogonalization is fast but has an unwarranted bad
# reputation: when applied twice to a matrix X with κ(X) <~
# sqrt(1/ε), where ε is the machine epsilon, it will produce a set
# of very orthogonal vectors, just as the eigendecomposition-based
# method. It can fail when κ >~ sqrt(1/ε), but that can be fixed by
# shifting the overlap matrix. This is very reliable while being
# much faster than eigendecompositions.

# - Termination criteria for the orthogonalization are based on cheap
# estimates instead of costly explicit checks.

# - The default tolerances are very tight, yielding a very stable
# algorithm. This can be tweaked (see keyword ortho_tol) to
# compromise on stability for less Cholesky factorizations.

# - Implicit product updates (reuse of matvecs) are performed whenever
# it is safe to do so, ie only with orthogonal transformations. An
# exception is the B matrix reuse, which seems to be OK even with very
# badly conditioned B matrices. The code is easy to modify if this is
# not the case.

# - The locking is performed carefully to ensure minimal impact on the
# other eigenvectors (which is not the case in many - all ? - other
# implementations)

# - Some functions are reimplemented in a GPU optimized way as part of
# the DFTK CUDA Extension (ext/DFTKCUDAExt/lobpcg.jl).


## TODO micro-optimization of buffer reuse
## TODO write a version that doesn't assume that B is well-conditioned, and doesn't reuse B applications at all
## TODO it seems there is a lack of orthogonalization immediately after locking, maybe investigate this to save on some Choleskys
## TODO debug orthogonalizations when A=I

using LinearAlgebra
import LinearAlgebra: BlasFloat
import Base: *
import Base.size, Base.adjoint, Base.Array

"""
Simple wrapper to represent a matrix formed by the concatenation of column blocks:
it is mostly equivalent to hcat, but doesn't allocate the full matrix.
LazyHcat only supports a few multiplication routines: furthermore, a multiplication
involving this structure will always yield a plain array (and not a LazyHcat structure).
LazyHcat is a lightweight subset of BlockArrays.jl's functionalities, but has the
advantage to be able to store GPU Arrays (BlockArrays is heavily built on Julia's CPU Array).
"""
struct LazyHcat{T<:Number, D<:Tuple} <: AbstractMatrix{T}
    blocks::D
end

function LazyHcat(arrays::AbstractArray...)
    @assert length(arrays) != 0
    n_ref = size(arrays[1], 1)
    @assert  all(size.(arrays, 1) .== n_ref)

    T = promote_type(map(eltype, arrays)...)

    LazyHcat{T, typeof(arrays)}(arrays)
end

function Base.size(A::LazyHcat)
    n = size(A.blocks[1], 1)
    m = sum(size(block, 2) for block in A.blocks)
    (n, m)
end

Base.Array(A::LazyHcat)   = stack(A.blocks)
Base.adjoint(A::LazyHcat) = Adjoint(A)

# Computes A*B matrix product for LazyHcat type. Special case if product is assumed to be Hermitian
@views function _mul(A::Adjoint{T,<:LazyHcat}, B::LazyHcat; hermitian=Val(false)) where {T}
    Ap = A.parent
    rows = size(Ap, 2)
    cols = size(B, 2)
    ret = similar(B.blocks[1], rows, cols)

    # Only popuplate the upper block diagonal in Hermitian case
    ocol = 0  # column offset
    for (ib, blB) in enumerate(B.blocks)
        orow = 0  # row offset
        for (ia, blA) in enumerate(Ap.blocks)
            (hermitian isa Val{true} && ib < ia) && continue
            ret[orow .+ (1:size(blA, 2)), ocol .+ (1:size(blB, 2))] .= blA' * blB
            orow += size(blA, 2)
        end
        ocol += size(blB, 2)
    end

    if hermitian isa Val{true}
        Hermitian(ret)
    else
        ret
    end
end

Base.:*(A::Adjoint{T,<:LazyHcat}, B::LazyHcat) where {T}       = _mul(A, B)
Base.:*(A::Adjoint{T,<:LazyHcat}, B::AbstractMatrix) where {T} = A * LazyHcat(B)

@views function *(Ablock::LazyHcat, B::AbstractMatrix)
    res = Ablock.blocks[1] * B[1:size(Ablock.blocks[1], 2), :]  # First multiplication
    offset = size(Ablock.blocks[1], 2)
    for block in Ablock.blocks[2:end]
        mul!(res, block, B[offset .+ (1:size(block, 2)), :], 1, 1)
        offset += size(block, 2)
    end
    res
end

function LinearAlgebra.mul!(res::AbstractMatrix, Ablock::LazyHcat,
                            B::AbstractVecOrMat, α::Number, β::Number)
    mul!(res, Ablock*B, I, α, β)
end

mul_hermi(A, B) = Hermitian(A * B)
function mul_hermi(A::Adjoint{T,<:LazyHcat}, B::LazyHcat) where {T}
    _mul(A, B; hermitian=Val(true))
end

# Perform a Rayleigh-Ritz for the N first eigenvectors.
@timing function rayleigh_ritz(X, AX, N)
    XAX = mul_hermi(X', AX)
    @assert !any(isnan, UpperTriangular(parent(XAX)))
    rayleigh_ritz(XAX, N)
end
@views function rayleigh_ritz(XAX::Hermitian, N)
    # Fallback: Use whatever is the default dense eigensolver.
    # Note: GenericLinearAlgebra uses a QR-based algorithm, which is pretty safe in terms
    #       of keeping the vectors orthogonal
    values, vectors = eigen(XAX)
    vectors[:, 1:N], values[1:N]
end
@views function rayleigh_ritz(XAX::Hermitian{<:BlasFloat, <:Array}, N)
    # LAPACK sysevr (the Julia default eigensolver up to 1.11 ) can actually return
    # eigenvectors that are significantly non-orthogonal (1e-4 in Float32 in some tests)
    # here, presumably because it tries hard to make them eigenvectors in the presence
    # of small gaps. syevd (or DivideAndConquer()) does a much better job, see
    # https://github.com/JuliaLang/julia/pull/49262 and
    # https://github.com/JuliaLang/julia/pull/49355. It will be the default in 1.12.
    # For versions < 1.12, since we mainly care about eigenvectors being orthogonal
    # we re-orthogonalise explicitly.
    @static if VERSION >= v"1.12"
        values, vectors = eigen(XAX; alg=DivideAndConquer())
        return vectors[:, 1:N], values[1:N]
    else
        values, vectors = eigen(XAX)
        v = vectors[:, 1:N]
        ortho!(v)
        return v, values[1:N]
    end
end

# B-orthogonalize X (in place) using only one B apply.
# This uses an unstable method which is only OK if X is already
# orthogonal (not B-orthogonal) and B is relatively well-conditioned
# (which implies that X'BX is relatively well-conditioned, and
# therefore that it is safe to cholesky it and reuse the B apply)
function B_ortho!(X, BX)
    O = mul_hermi(X', BX)
    U = cholesky(O).U
    @assert !any(isnan, U)
    rdiv!(X, U)
    rdiv!(BX, U)
end

normest(M) = maximum(abs, diag(M)) + norm(M - Diagonal(diag(M)))
# Orthogonalizes X to tol
# Returns the new X, the number of Cholesky factorizations algorithm, and the
# growth factor by which small perturbations of X can have been magnified
@timing function ortho!(X::AbstractArray{T}; tol=2eps(real(T))) where {T}
    local R
    growth_factor = one(real(T))

    success = false
    nchol = 0
    while true
        O = mul_hermi(X', X)
        try
            R = cholesky(O).U
            nchol += 1
            success = true
        catch err
            @assert isa(err, PosDefException)
            @debug "Cholesky failed in ortho(X)"
            # see https://arxiv.org/pdf/1809.11085.pdf for a nice analysis
            # We are not being very clever here; but this should very rarely happen
            # so it should be OK
            α = 100
            nbad = 0
            while true
                O += α*eps(real(T))*norm(X)^2*I
                α *= 10
                try
                    R = cholesky(O).U
                    nchol += 1
                    break
                catch err
                    @assert isa(err, PosDefException)
                end
                nbad += 1
                if nbad > 10
                    @error "Cholesky shifting is failing badly, falling back to SVD"
                    U, _, V = svd(X)
                    return (; X=U*V', nchol=1000, growth_factor=1)
                end
            end
            success = false
        end
        invR = inv(R)
        @assert !any(isnan, invR)
        rmul!(X, invR)  # we do not use X/R because we use invR next

        # We would like growth_factor *= opnorm(inv(R)) but it's too
        # expensive, so we use an upper bound which is sharp enough to
        # be close to 1 when R is close to I, which we need for the
        # outer loop to terminate
        # ||invR|| = ||D + E|| ≤ ||D|| + ||E|| ≤ ||D|| + ||E||_F,
        # where ||.|| is the 2-norm and ||.||_F the Frobenius

        norminvR = normest(invR)
        growth_factor *= norminvR

        # condR = 1/LAPACK.trcon!('I', 'U', 'N', Array(R))
        condR = normest(R)*norminvR  # in practice this seems to be an OK estimate

        @debug "Ortho(X) success? $success : $(eps(real(T))*condR^2) < $tol"

        # a good a posteriori error is that X'X - I is eps()*κ(R)^2;
        # in practice this seems to be sometimes very overconservative
        estimated_error = eps(real(T))*condR^2
        success && estimated_error < tol && break

        if nchol > 10
            @error("Ortho(X) is failing badly, falling back to SVD",
                   estimated_error=round(estimated_error; sigdigits=2), tol=tol)
            U, _, V = svd(X)
            return (; X=U*V', nchol=100, growth_factor=1)
        end
    end

    # @assert norm(X'X - I) < tol

    (; X, nchol, growth_factor)
end

# Randomize the columns of X if the norm is below tol
function drop_small!(X::AbstractArray{T}; tol=2eps(real(T))) where {T}
    dropped = findall(n -> n <= tol, columnwise_norms(X))
    @views randn!(TaskLocalRNG(), X[:, dropped])
    dropped
end

# Find X that is orthogonal, and B-orthogonal to Y, up to a tolerance tol.
@timing "ortho! X vs Y" function ortho!(X::AbstractArray{T}, Y, BY; tol=2eps(real(T))) where {T}
    # normalize to try to cheaply improve conditioning
    X ./= columnwise_norms(X)'

    niter = 1
    ninners = zeros(Int, 0)
    while true
        BYX = BY' * X
        mul!(X, Y, BYX, -1, 1)  # X -= Y*BY'X
        # If the orthogonalization has produced results below 2eps, we drop them
        # This is to be able to orthogonalize eg [1;0] against [e^iθ;0],
        # as can happen in extreme cases in the ortho!(cP, cX)
        dropped = drop_small!(X; tol)
        if !isempty(dropped)
            X[:, dropped] .-= Y * (BY' * X[:, dropped])
        end

        if norm(BYX) < tol && niter > 1
            push!(ninners, 0)
            break
        end
        X, ninner, growth_factor = ortho!(X; tol)
        push!(ninners, ninner)

        # norm(BY'X) < tol && break should be the proper check, but
        # usually unnecessarily costly. Instead, we predict the error
        # according to the growth factor of the orthogonalization.
        # Assuming BY'Y = 0 to machine precision, after the
        # Y-orthogonalization, BY'X is O(eps), so BY'X will be bounded
        # by O(eps * growth_factor).

        # If we're at a fixed point, growth_factor is 1 and if tol > eps(),
        # the loop will terminate, even if BY'Y != 0
        estimated_error = growth_factor * eps(real(T))
        estimated_error < tol && break

        if niter > 10
            U, _, V = svd(X)  # Fall back to gold standard
            X = U*V'
            @error("Ortho(X, Y) is failing badly, falling back to SVD",
                   ninners=ninners, error=round(norm(BY'X); sigdigits=2), tol=tol,
                   estimated_error=round(estimated_error; sigdigits=2))
            return X
        end
        niter += 1
    end
    @debug "Required $ninners choleskys in ortho!(X, Y)"

    # @assert (norm(BY'X)) < tol
    # @assert (norm(X'X-I)) < tol

    X
end

function final_retval(X, AX, BX, λ, resid_history, niter, n_matvec)
    λ_host = to_cpu(λ)  # Copy to CPU for element-wise access
    if !issorted(λ_host)
        p = sortperm(λ_host)
        λ_host = λ_host[p]
        X  = X[:, p]
        AX = AX[:, p]
        BX = BX[:, p]
        resid_history = resid_history[p, :]
    end
    (; λ=λ_host, X, AX, BX,
     residual_norms=resid_history[:, niter+1],
     residual_history=resid_history[:, 1:niter+1], n_matvec)
end

# Computes λ = real((X' * AX) / (X' *BX)), for each column of X
function compute_λ(X, AX, BX)
    λs = @views [real((X[:, n]'*AX[:, n]) / (X[:, n]'BX[:, n])) for n=1:size(X, 2)]
    oftype(real(X[:, 1]), λs)  # Offload to GPU if needed
end

### The algorithm is Xn+1 = rayleigh_ritz(hcat(Xn, A*Xn, Xn-Xn-1))
### We follow the strategy of Hetmaniuk and Lehoucq, and maintain a B-orthonormal basis Y = (X,R,P)
### After each rayleigh_ritz step, the B-orthonormal X and P are deduced by an orthogonal rotation from Y
### R is then recomputed, and orthonormalized explicitly wrt BX and BP
### We reuse applications of A/B when it is safe to do so, ie only with orthogonal transformations

# Note that this function will return λ on the CPU,
# but X and the history on the device (for GPU runs)
@timing function LOBPCG(A, X, B=I, precon=I, tol=1e-10, maxiter=100;
                        miniter=1, ortho_tol=2eps(real(eltype(X))),
                        n_conv_check=nothing, display_progress=false)
    N, M = size(X)

    # If N is too small, we will likely get in trouble
    error_message(verb) = "The eigenproblem is too small, and the iterative " *
                           "eigensolver $verb fail; increase the number of " *
                           "degrees of freedom, or use a dense eigensolver."
    N > 3M    || error(error_message("will"))
    N >= 3M+5 || @warn error_message("might")

    isnothing(n_conv_check) && (n_conv_check = M)
    resid_history = zeros(real(eltype(X)), M, maxiter+1)

    # B-orthogonalize X
    X = ortho!(copy(X), tol=ortho_tol)[1]
    if B != I
        BX = similar(X)
        BX = mul!(BX, B, X)
        B_ortho!(X, BX)
    end

    n_matvec = M  # Count number of matrix-vector products
    AX = similar(X)
    AX = mul!(AX, A, X)
    @assert !any(isnan, AX)
    # full_X/AX/BX will always store the full (including locked) X.
    # X/AX/BX only point to the active part
    P = zero(X)
    AP = zero(X)
    R = zero(X)
    AR = zero(X)
    if B != I
        BR = zero(X)
        # BX was already computed
        BP = zero(X)
    else
        # The B arrays are just pointers to the same data
        BR = R
        BX = X
        BP = P
    end
    nlocked = 0
    niter = 0  # the first iteration is fake
    λs = compute_λ(X, AX, BX)
    new_X  = X
    new_AX = AX
    new_BX = BX
    # The full_ arrays contain all the vectors, the others only get the active ones
    full_X  = X
    full_AX = AX
    full_BX = BX
    full_λs = λs

    while true
        if niter > 0  # first iteration is just to compute the residuals (no X update)
            ###  Perform the Rayleigh-Ritz
            mul!(AR, A, R)
            n_matvec += size(R, 2)

            # Form Rayleigh-Ritz subspace
            if niter > 1
                Y  = LazyHcat(X, R, P)
                AY = LazyHcat(AX, AR, AP)
                BY = LazyHcat(BX, BR, BP)  # data shared with (X, R, P) in non-general case
            else
                Y  = LazyHcat(X, R)
                AY = LazyHcat(AX, AR)
                BY = LazyHcat(BX, BR)  # data shared with (X, R) in non-general case
            end
            cX, λs_RR = rayleigh_ritz(Y, AY, M-nlocked)
            λs .= λs_RR

            # Update X. By contrast to some other implementations, we
            # wait on updating P because we have to know which vectors
            # to lock (and therefore the residuals) before computing P
            # only for the unlocked vectors. This results in better convergence.
            new_X  = Y * cX
            new_AX = AY * cX  # no accuracy loss, since cX orthogonal
            new_BX = (B == I) ? new_X : BY * cX
        end

        ### Compute new residuals
        @timing "Update residuals" begin
            new_R = new_AX .- new_BX .* λs'
            norms = to_cpu(columnwise_norms(new_R))
            @views resid_history[1 + nlocked: size(new_R, 2) + nlocked, niter+1] .= norms[:]
        end
        @debug niter resid_history[:, niter+1]

        # it is actually a good question of knowing when to
        # precondition. Here seems sensible, but it could plausibly be
        # done before or after
        if precon !== I
            @timing "preconditioning" begin
                precondprep!(precon, X)  # update preconditioner if needed; defaults to noop
                ldiv!(precon, new_R)
            end
        end

        ### Compute number of locked vectors
        prev_nlocked = nlocked
        if niter ≥ miniter  # No locking if below miniter
            for i=nlocked+1:M
                if resid_history[i, niter+1] < tol
                    nlocked += 1
                    @debug "Locked eigenvector $nlocked"
                else
                    # We lock in order, assuming that the lowest
                    # eigenvectors converge first; might be tricky otherwise
                    break
                end
            end
        end

        if display_progress
            println("Iter $niter, converged $(nlocked)/$(n_conv_check), resid ",
                    norm(resid_history[1:n_conv_check, niter+1]))
        end

        if nlocked >= n_conv_check  # Converged!
            X  .= new_X  # Update the part of X which is still active
            AX .= new_AX
            return final_retval(full_X, full_AX, full_BX, full_λs, resid_history, niter, n_matvec)
        end
        newly_locked = nlocked - prev_nlocked
        active = newly_locked+1:size(X,2)  # newly active vectors

        if niter > 0
            ### compute P = Y*cP only for the newly active vectors
            Xn_indices = newly_locked+1:M-prev_nlocked
            # TODO understand this for a potential save of an
            # orthogonalization, see Hetmaniuk & Lehoucq, and Duersch et. al.
            # cP = copy(cX)
            # cP[Xn_indices,:] .= 0

            lenXn = length(Xn_indices)
            e = zeros_like(X, size(cX, 1), M - prev_nlocked)
            lower_diag = one(similar(X, lenXn, lenXn))
            # e has zeros everywhere except on one of its lower diagonal
            e[Xn_indices[1]:last(Xn_indices), 1:lenXn] = lower_diag

            cP = cX .- e
            cP = cP[:, Xn_indices]
            # orthogonalize against all Xn (including newly locked)
            ortho!(cP, cX, cX, tol=ortho_tol)

            # Get new P
            new_P  = Y  * cP
            new_AP = AY * cP
            if B != I
                new_BP = BY * cP
            else
                new_BP = new_P
            end
        end

        # Update all X, even newly locked
        X  .= new_X
        AX .= new_AX
        if B != I
            BX .= new_BX
        end

        # Quick sanity check
        diffs = abs.(columnwise_dots(BX, X) .-1)
        if any(diffs .>= sqrt(eps(real(eltype(X)))))
           error("LOBPCG is badly failing to keep the vectors normalized; this should never happen")
        end

        # Restrict all views to active
        @views begin
            X = X[:, active]
            AX = AX[:, active]
            BX = BX[:, active]
            R = new_R[:, active]
            AR = AR[:, active]
            BR = BR[:, active]
            P = P[:, active]
            AP = AP[:, active]
            BP = BP[:, active]
            λs = λs[active]
        end

        # Update newly active P
        if niter > 0
            P .= new_P
            AP .= new_AP
            if B != I
                BP .= new_BP
            end
        end

        # Orthogonalize R wrt all X, newly active P
        if niter > 0
            Z  = LazyHcat(full_X, P)
            BZ = LazyHcat(full_BX, BP)  # data shared with (full_X, P) in non-general case
        else
            Z  = full_X
            BZ = full_BX
        end
        ortho!(R, Z, BZ; tol=ortho_tol)
        if B != I
            mul!(BR, B, R)
            B_ortho!(R, BR)
        end

        niter >= maxiter && break
        niter = niter + 1
    end

    final_retval(full_X, full_AX, full_BX, full_λs, resid_history, maxiter, n_matvec)
end
