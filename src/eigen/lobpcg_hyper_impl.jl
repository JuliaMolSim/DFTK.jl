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


## TODO micro-optimization of buffer reuse
## TODO write a version that doesn't assume that B is well-conditionned, and doesn't reuse B applications at all
## TODO it seems there is a lack of orthogonalization immediately after locking, maybe investigate this to save on some choleskys
## TODO debug orthogonalizations when A=I

# vprintln(args...) = println(args...)  # Uncomment for output
vprintln(args...) = nothing

using LinearAlgebra
import Base: *
import Base.size
include("../workarounds/gpu_arrays.jl")

# For now, BlockMatrix can store arrays of different types (for example, an element 
# of type views and one of type Matrix). Maybe for performance issues it should only
# store arrays of the same type?

struct BlockMatrix{T <: Number, D <: Tuple} <: AbstractMatrix{T}
    blocks::D
end

"""
Build a BlockMatrix containing the given arrays, from left to right.
This function will fail (for now) if:
    -the arrays do not all have the same "height" (ie size[1] must match).
"""
function BlockMatrix(arrays::AbstractArray...)
    length(arrays) ==0 && error("Empty BlockMatrix is not currently implemented")
    n_ref= size(arrays[1],1)
    for array in arrays
        n_i = size(array, 1)
        n_ref != n_i && error("The given arrays do not have matching 'height': "*
        "cannot build a BlockMatrix out of them.")
    end

    T = promote_type(map(eltype, arrays)...)

    BlockMatrix{T, typeof(arrays)}(arrays)
end

function Base.size(A::BlockMatrix)
    n = size(A.blocks[1],1)
    m = sum(size(block,2) for block in A.blocks)
    (n,m)
end
"""
Given A and B as two BlockMatrixs [A1, A2, A3], [B1, B2, B3] form the matrix
A'B (which is not a BlockMatrix). block_overlap also has compatible versions with two Arrays. 
block_overlap always compute some form of adjoint, ie the product A'*B.
"""
@views function block_overlap(A::BlockMatrix, B::BlockMatrix)
    rows = size(A)[2]
    cols = size(B)[2]
    ret = similar(A.blocks[1], rows, cols)

    orow = 0  # row offset
    for (iA, blA) in enumerate(A.blocks)
        ocol = 0  # column offset
        for (iB, blB) in enumerate(B.blocks)
            ret[orow .+ (1:size(blA, 2)), ocol .+ (1:size(blB, 2))] = blA' * blB
            ocol += size(blB, 2)
        end
        orow += size(blA, 2)
    end
    ret
end

block_overlap(blocksA::BlockMatrix, B) = block_overlap(blocksA, BlockMatrix(B))
block_overlap(A, B) = A' * B  # Default fallback method. Note the adjoint.

"""
Given A as a BlockMatrix [A1, A2, A3] and B a Matrix, compute the matrix-matrix product
A * B avoiding a concatenation of the blocks to a dense array. 
"""
@views function *(Ablock::BlockMatrix, B::AbstractMatrix)
    res = Ablock.blocks[1] * B[1:size(Ablock.blocks[1], 2), :]  # First multiplication
    offset = size(Ablock.blocks[1], 2)
    for block in Ablock.blocks[2:end]
        mul!(res, block, B[offset .+ (1:size(block, 2)), :], 1, 1)
        offset += size(block, 2)
    end
    res
end

function LinearAlgebra.mul!(res::AbstractMatrix,A::BlockMatrix,B::AbstractVecOrMat,α::Number,β::Number)
    # Has slightly better performances than a naive res = α*A*B - β*res
    mul!(res, A*B, I, α, β)
end

# Perform a Rayleigh-Ritz for the N first eigenvectors.
@timing function rayleigh_ritz(X::BlockMatrix, AX::BlockMatrix, N)
    # block_overlap(X,AX) is an AbstractArray, not a BlockMatrix
    F = eigen(Hermitian(block_overlap(X, AX)))
    F.vectors[:,1:N], F.values[1:N]
end

# B-orthogonalize X (in place) using only one B apply.
# This uses an unstable method which is only OK if X is already
# orthogonal (not B-orthogonal) and B is relatively well-conditioned
# (which implies that X'BX is relatively well-conditioned, and
# therefore that it is safe to cholesky it and reuse the B apply)
function B_ortho!(X, BX)
    O = Hermitian(X'*BX)
    U = cholesky(O).U
    rdiv!(X, U)
    rdiv!(BX, U)
end

normest(M) = maximum(abs.(diag(M))) + norm(M - Diagonal(diag(M)))
# Orthogonalizes X to tol
# Returns the new X, the number of Cholesky factorizations algorithm, and the
# growth factor by which small perturbations of X can have been
# magnified
@timing function ortho!(X::AbstractArray{T}; tol=2eps(real(T))) where T
    local R

    # # Uncomment for "gold standard"
    # U,S,V = svd(X)
    # return U*V', 1, 1

    growth_factor = 1

    success = false
    nchol = 0
    while true
        O = Hermitian(X'X)
        try
            R = cholesky(O).U
            nchol += 1
            success = true
        catch err
            @assert isa(err, PosDefException)
            vprintln("fail")
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
                    error("Cholesky shifting is failing badly, this should never happen")
                end
            end
            success = false
        end
        invR = inv(R)
        rmul!(X, invR) # we do not use X/R because we use invR next

        # We would like growth_factor *= opnorm(inv(R)) but it's too
        # expensive, so we use an upper bound which is sharp enough to
        # be close to 1 when R is close to I, which we need for the
        # outer loop to terminate
        # ||invR|| = ||D + E|| ≤ ||D|| + ||E|| ≤ ||D|| + ||E||_F,
        # where ||.|| is the 2-norm and ||.||_F the Frobenius

        norminvR = normest(invR)
        growth_factor *= norminvR

        # condR = 1/LAPACK.trcon!('I', 'U', 'N', Array(R))
        condR = normest(R)*norminvR # in practice this seems to be an OK estimate

        vprintln("Ortho(X) success? $success ", eps(real(T))*condR^2, " < $tol")

        # a good a posteriori error is that X'X - I is eps()*κ(R)^2;
        # in practice this seems to be sometimes very overconservative
        success && eps(real(T))*condR^2 < tol && break

        nchol > 10 && error("Ortho(X) is failing badly, this should never happen")
    end

    # @assert norm(X'X - I) < tol

    X, nchol, growth_factor
end

# Randomize the columns of X if the norm is below tol
function drop!(X::AbstractArray{T}, tol=2eps(real(T))) where T
    dropped = Int[]
    for i=1:size(X,2)
        n = norm(@views X[:,i])
        if n <= tol
            X[:,i] = randn(T, size(X,1))
            push!(dropped, i)
        end
    end
    dropped
end

# Find X that is orthogonal, and B-orthogonal to Y, up to a tolerance tol.
@timing "ortho! X vs Y" function ortho!(X::AbstractArray{T}, Y, BY; tol=2eps(real(T))) where T
    # normalize to try to cheaply improve conditioning
    Threads.@threads for i=1:size(X,2)
        n = norm(@views X[:,i])
        @views X[:,i] ./= n
    end

    niter = 1
    ninners = zeros(Int,0)
    while true
        BYX = block_overlap(BY,X)  # = BY' X
        mul!(X, Y, BYX, -one(T), one(T))  # X -= Y*BY'X
        # If the orthogonalization has produced results below 2eps, we drop them
        # This is to be able to orthogonalize eg [1;0] against [e^iθ;0],
        # as can happen in extreme cases in the ortho!(cP, cX)
        dropped = drop!(X)
        if dropped != []
            X[:, dropped] .-= Y * block_overlap(BY,X[:, dropped])  # X = X - Y'*BY*X
        end

        if norm(BYX) < tol && niter > 1
            push!(ninners, 0)
            break
        end
        X, ninner, growth_factor = ortho!(X, tol=tol)
        push!(ninners, ninner)

        # norm(BY'X) < tol && break should be the proper check, but
        # usually unnecessarily costly. Instead, we predict the error
        # according to the growth factor of the orthogonalization.
        # Assuming BY'Y = 0 to machine precision, after the
        # Y-orthogonalization, BY'X is O(eps), so BY'X will be bounded
        # by O(eps * growth_factor).

        # If we're at a fixed point, growth_factor is 1 and if tol >
        # eps(), the loop will terminate, even if BY'Y != 0
        growth_factor*eps(real(T)) < tol && break

        niter > 10 && error("Ortho(X,Y) is failing badly, this should never happen")
        niter += 1
    end
    vprintln("ortho choleskys: ", ninners) # get how many Choleskys are performed

    # @assert (norm(BY'X)) < tol
    # @assert (norm(X'X-I)) < tol

    X
end


function final_retval(X, AX, resid_history, niter, n_matvec)
    λ = real(diag(X' * AX))
    residuals = AX .- X*Diagonal(λ)
    (λ=λ, X=X,
     residual_norms=[norm(residuals[:, i]) for i in 1:size(residuals, 2)],
     residual_history=resid_history[:, 1:niter+1], n_matvec=n_matvec)
end

### The algorithm is Xn+1 = rayleigh_ritz(hcat(Xn, A*Xn, Xn-Xn-1))
### We follow the strategy of Hetmaniuk and Lehoucq, and maintain a B-orthonormal basis Y = (X,R,P)
### After each rayleigh_ritz step, the B-orthonormal X and P are deduced by an orthogonal rotation from Y
### R is then recomputed, and orthonormalized explicitly wrt BX and BP
### We reuse applications of A/B when it is safe to do so, ie only with orthogonal transformations

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

    n_conv_check === nothing && (n_conv_check = M)
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
    @assert all(!isnan, AX)
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
    λs = @views [(X[:,n]'*AX[:,n]) / (X[:,n]'BX[:,n]) for n=1:M]
    λs = oftype(X[:,1], λs)  # Offload to GPU if needed
    new_X = X
    new_AX = AX
    new_BX = BX
    full_X = X
    full_AX = AX
    full_BX = BX

    while true
        if niter > 0 # first iteration is just to compute the residuals (no X update)
            ###  Perform the Rayleigh-Ritz
            mul!(AR, A, R)
            n_matvec += size(R, 2)

            # Form Rayleigh-Ritz subspace
            if niter > 1
                Y = BlockMatrix(X, R, P)
                AY = BlockMatrix(AX, AR, AP)
                BY = BlockMatrix(BX, BR, BP)  # data shared with (X, R, P) in non-general case
            else
                Y  = BlockMatrix(X, R)
                AY = BlockMatrix(AX, AR)
                BY = BlockMatrix(BX, BR)  # data shared with (X, R) in non-general case
            end
            cX, λs = rayleigh_ritz(Y, AY, M-nlocked)

            # Update X. By contrast to some other implementations, we
            # wait on updating P because we have to know which vectors
            # to lock (and therefore the residuals) before computing P
            # only for the unlocked vectors. This results in better convergence.
            new_X  = Y * cX
            new_AX = AY * cX  # no accuracy loss, since cX orthogonal
            new_BX = (B == I) ? new_X : BY * cX
        end

        ### Compute new residuals
        new_R = new_AX .- new_BX .* λs'
        # it is actually a good question of knowing when to
        # precondition. Here seems sensible, but it could plausibly be
        # done before or after
        @views for i=1:size(X,2)
            resid_history[i + nlocked, niter+1] = norm(new_R[:, i])
        end
        vprintln(niter, "   ", resid_history[:, niter+1])
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
                    vprintln("locked $nlocked")
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
            X .= new_X  # Update the part of X which is still active
            AX .= new_AX
            return final_retval(full_X, full_AX, resid_history, niter, n_matvec)
        end
        newly_locked = nlocked - prev_nlocked
        active = newly_locked+1:size(X,2) # newly active vectors

        if niter > 0
            ### compute P = Y*cP only for the newly active vectors
            Xn_indices = newly_locked+1:M-prev_nlocked
            # TODO understand this for a potential save of an
            # orthogonalization, see Hetmaniuk & Lehoucq, and Duersch et. al.
            # cP = copy(cX)
            # cP[Xn_indices,:] .= 0

            lenXn = length(Xn_indices)
            e = zero(similar(X, size(cX, 1), M - prev_nlocked))
            lower_diag = one(similar(X, lenXn, lenXn))
            # e has zeros everywhere except on one of its lower diagonal
            e[Xn_indices[1] : last(Xn_indices), 1 : lenXn] = lower_diag

            cP = cX .- e
            cP = cP[:, Xn_indices]
            # orthogonalize against all Xn (including newly locked)
            ortho!(cP, cX, cX, tol=ortho_tol)

            # Get new P
            new_P  = Y * cP
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
        for i = 1:size(X, 2)
            @views if abs(BX[:, i]'X[:, i] - 1) >= sqrt(eps(real(eltype(X))))
                error("LOBPCG is badly failing to keep the vectors normalized; this should never happen")
            end
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
            Z  = BlockMatrix(full_X, P)
            BZ = BlockMatrix(full_BX, BP) # data shared with (full_X, P) in non-general case
        else
            Z  = full_X
            BZ = full_BX
        end
        ortho!(R, Z, BZ; tol=ortho_tol)
        if B != I
            mul!(BR, B, R)
            B_ortho!(R, BR)
        end

        niter < maxiter || break
        niter = niter + 1
    end

    final_retval(full_X, full_AX, resid_history, maxiter, n_matvec)
end
