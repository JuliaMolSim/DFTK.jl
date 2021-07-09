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
using BlockArrays # used for the `mortar` command which makes block matrices

# when X or Y are BlockArrays, this makes the return value be a proper array (not a BlockArray)
function array_mul(X, Y)
    Z = zeros(eltype(X), size(X, 1), size(Y, 2))
    mul!(Z, X, Y)
end

# Perform a Rayleigh-Ritz for the N first eigenvectors.
@timing function rayleigh_ritz(X, AX, N)
    F = eigen(Hermitian(array_mul(X', AX)))
    F.vectors[:,1:N], F.values[1:N]
end

import LinearAlgebra: cholesky
function cholesky(X::Union{Matrix{ComplexF16}, Hermitian{ComplexF16,Matrix{ComplexF16}}})
    # Cholesky factorization above may promote the type
    # (e.g. Float16 is promoted to Float32. This undoes it)
    # See https://github.com/JuliaLang/julia/issues/16446
    U = cholesky(ComplexF32.(X)).U
    (U=convert.(ComplexF16, U), )
end

# B-orthogonalize X (in place) using only one B apply.
# This uses an unstable method which is only OK if X is already
# orthogonal (not B-orthogonal) and B is relatively well-conditioned
# (which implies that X'BX is relatively well-conditioned, and
# therefore that it is safe to cholesky it and reuse the B aply)
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
@timing function ortho!(X; tol=2eps(real(eltype(X))))
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
                O += α*eps(real(eltype(X)))*norm(X)^2*I
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

        vprintln("Ortho(X) success? $success ", eps(real(eltype(X)))*condR^2, " < $tol")

        # a good a posteriori error is that X'X - I is eps()*κ(R)^2;
        # in practice this seems to be sometimes very overconservative
        success && eps(real(eltype(X)))*condR^2 < tol && break

        nchol > 10 && error("Ortho(X) is failing badly, this should never happen")
    end

    # @assert norm(X'X - I) < tol

    X, nchol, growth_factor
end

# Randomize the columns of X if the norm is below tol
function drop!(X, tol=2eps(real(eltype(X))))
    dropped = Int[]
    for i=1:size(X,2)
        n = norm(@views X[:,i])
        if n <= tol
            X[:,i] = randn(eltype(X), size(X,1))
            push!(dropped, i)
        end
    end
    dropped
end

# Find X that is orthogonal, and B-orthogonal to Y, up to a tolerance tol.
function ortho!(X, Y, BY; tol=2eps(real(eltype(X))))
    T = real(eltype(X))
    # normalize to try to cheaply improve conditioning
    Threads.@threads for i=1:size(X,2)
        n = norm(@views X[:,i])
        @views X[:,i] ./= n
    end

    niter = 1
    ninners = zeros(Int,0)
    while true
        BYX = BY'X
        # XXX the one(T) instead of plain old 1 is because of https://github.com/JuliaArrays/BlockArrays.jl/issues/176
        mul!(X, Y, BYX, -one(T), one(T)) # X -= Y*BY'X
        # If the orthogonalization has produced results below 2eps, we drop them
        # This is to be able to orthogonalize eg [1;0] against [e^iθ;0],
        # as can happen in extreme cases in the ortho!(cP, cX)
        dropped = drop!(X)
        if dropped != []
            @views mul!(X[:, dropped], Y, BY' * (X[:, dropped]), -one(T), one(T)) # X -= Y*BY'X
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
        growth_factor*eps(real(eltype(X))) < tol && break

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
     residual_history=resid_history[:, 1:niter+1],
     n_matvec=n_matvec)
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
    N >= 3M || @warn "Your problem is too small, and LOBPCG might
        fail; use a full diagonalization instead"

    n_conv_check === nothing && (n_conv_check = M)
    resid_history = zeros(real(eltype(X)), M, maxiter+1)

    # B-orthogonalize X
    X = ortho!(copy(X), tol=ortho_tol)[1]
    if B != I
        BX = similar(X)
        BX = mul!(BX, B, X)
        B_ortho!(X, BX)
    end

    n_matvec = M   # Count number of matrix-vector products
    AX = similar(X)
    AX = mul!(AX, A, X)
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
                Y = mortar((X, R, P))
                AY = mortar((AX, AR, AP))
                BY = mortar((BX, BR, BP))  # data shared with (X, R, P) in non-general case
            else
                Y  = mortar((X, R))
                AY = mortar((AX, AR))
                BY = mortar((BX, BR))  # data shared with (X, R) in non-general case
            end
            cX, λs = rayleigh_ritz(Y, AY, M-nlocked)

            # Update X. By contrast to some other implementations, we
            # wait on updating P because we have to know which vectors
            # to lock (and therefore the residuals) before computing P
            # only for the unlocked vectors. This results in better convergence.
            new_X  = array_mul(Y, cX)
            new_AX = array_mul(AY, cX)  # no accuracy loss, since cX orthogonal
            new_BX = (B == I) ? new_X : array_mul(BY, cX)
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
            precondprep!(precon, X) # update preconditioner if needed; defaults to noop
            ldiv!(precon, new_R)
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
            e = zeros(eltype(X), size(cX, 1), M - prev_nlocked)
            for i in 1:length(Xn_indices)
                e[Xn_indices[i], i] = 1
            end
            cP = cX .- e
            cP = cP[:, Xn_indices]
            # orthogonalize against all Xn (including newly locked)
            ortho!(cP, cX, cX, tol=ortho_tol)

            # Get new P
            new_P  = array_mul( Y, cP)
            new_AP = array_mul(AY, cP)
            if B != I
                new_BP = array_mul(BY, cP)
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
            Z  = mortar((full_X, P))
            BZ = mortar((full_BX, BP))  # data shared with (full_X, P) in non-general case
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
