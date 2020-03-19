## TODO micro-optimization, reuse buffers, etc.
## TODO hcat -> Hcat if/when https://github.com/JuliaArrays/LazyArrays.jl/issues/48 is fixed
## TODO write a version that doesn't assume that B is well-conditionned, and doesn't reuse B applications
## TODO it seems there is a lack of orthogonalization immediately after locking, maybe investigate this to save on some choleskys

# vprintln(args...) = println(args...)  # Uncomment for output
vprintln(args...) = nothing


using LinearAlgebra
using GenericLinearAlgebra

# Perform a Rayleigh-Ritz for the N first eigenvectors.
function RR(X, AX, BX, N)
    F = eigen(Hermitian(X'AX))
    F.vectors[:,1:N], F.values[1:N]
end

import LinearAlgebra: cholesky
function cholesky(X::Union{Matrix{ComplexF16}, Hermitian{ComplexF16,Matrix{ComplexF16}}})
    # Cholesky factorisation above may promote the type
    # (e.g. Float16 is promoted to Float32. This undoes it)
    # See https://github.com/JuliaLang/julia/issues/16446
    U = cholesky(ComplexF32.(X)).U
    (U=convert.(ComplexF16, U), )
end

# Orthogonalizes X to tol
# Returns the new X, the number of Cholesky factorizations algorithm, and the
# growth factor by which small perturbations of X can have been
# magnified
function ortho(X; tol=2eps(real(eltype(X))))
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
        X = X*invR # we do not use X/R because we use invR next

        normest(M) = maximum(abs.(diag(M))) + norm(M - Diagonal(diag(M)))

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
    dropped = []
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
function ortho(X, Y, BY; tol=2eps(real(eltype(X))))
    for i=1:size(X,2)
        n = norm(@views X[:,i])
        X[:,i] /= n
    end

    niter = 1
    ninners = zeros(Int,0)
    while true
        BYX = BY'X
        X .-= Y*BYX
        # If the orthogonalization has produced results below 2eps, we drop them
        # This is to be able to orthogonalize eg [1;0] against [e^iθ;0],
        # as can happen in extreme cases in the ortho(cP, cX)
        dropped = drop!(X)
        if dropped != []
            X[:, dropped] = X[:, dropped] - Y*BY'*X[:,dropped]
        end
        if norm(BYX) < tol && niter > 1
            push!(ninners, 0)
            break
        end
        X, ninner, growth_factor = ortho(X, tol=tol)
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


function final_residnorms(A, X, resids, niter)
    AX = A * X
    λ = real(diag(X' * AX))
    residuals = AX - X*Diagonal(λ)
    λ, X, [norm(residuals[:, i]) for i in 1:size(residuals, 2)], resids[:, 1:niter]
end


### The algorithm is Xn+1 = RR(hcat(Xn, A*Xn, Xn-Xn-1))
### We follow the strategy of Hetmaniuk and Lehoucq, and maintain a B-orthonormal basis Y = (X,R,P)
### After each RR step, the B-orthonormal X and P are deduced by an orthogonal rotation from Y
### R is then recomputed, and orthonormalized explicitly wrt BX and BP
### We reuse applications of A/B when it is safe to do so, ie only with orthogonal transformations

function LOBPCG(A, X, B=I, precon=((Y, X, R)->R), tol=1e-10, maxiter=100; ortho_tol=2eps(real(eltype(X))),
                n_conv_check=nothing, display_progress=false)
    N,M = size(X)

    # If N is too small, we will likely get in trouble
    N >= 3M || @warn "Your problem is too small, and LOBPCG might
        fail; use a full diagonalization instead"

    n_conv_check === nothing && (n_conv_check = M)
    resids = zeros(real(eltype(X)), M, maxiter)
    buf_X = zero(X)
    buf_P = zero(X)

    X = ortho(X, tol=ortho_tol)[1]
    AX = A*X
    # full_X/AX/BX will always store the full (including locked) X.
    # X/AX/BX only point to the active part
    P = zero(X)
    AP = zero(X)
    R = zero(X)
    AR = zero(X)
    if B != I
        BR = zero(X)
        BX = B*X
        BP = zero(X)
    else
        # The B arrays share the data
        BR = R
        BX = X
        BP = P
    end
    nlocked = 0
    niter = 1
    λs = [(X[:,n]'*AX[:,n]) / (X[:,n]'BX[:,n]) for n=1:M]
    new_X = X
    new_AX = AX
    new_BX = BX
    full_X = X
    full_AX = AX
    full_BX = BX

    while true
        if niter > 1 # first iteration is just to compute the residuals
            ###  Perform the Rayleigh-Ritz
            mul!(AR, A, R)

            # Form Rayleigh-Ritz subspace
            if niter > 2
                Y = hcat(X,R,P)
                AY = hcat(AX,AR,AP)
                BY = hcat(BX,BR,BP)
            else
                Y = hcat(X,R)
                AY = hcat(AX,AR)
                BY = hcat(BX,BR)
            end


            cX, λs = RR(Y, AY, BY, M-nlocked)

            # Update X. By contrast to some other implementations, we
            # wait on updating P because we have to know which vectors
            # to lock (and therefore the residuals) before computing P
            # only for the unlocked vectors. This results in better
            # convergence.
            new_X = Y*cX
            # cX is orthogonal, so there is no accuracy loss there
            new_AX = AY*cX
            if B != I
                new_BX = BY*cX
            else
                new_BX = new_X
            end
        end

        ### Compute new residuals
        R = new_AX - new_BX*Diagonal(λs)
        # it is actually a good question of knowing when to
        # precondition. Here seems sensible, but it could plausibly be
        # done before or after
        for i=1:size(X,2)
            resids[i+nlocked,niter] = norm(R[:,i])
        end
        vprintln(niter, "   ", resids[:, niter])
        precondprep!(precon, X)
        ldiv!(precon, R)

        ### Compute number of locked vectors
        prev_nlocked = nlocked
        for i=nlocked+1:M
            if resids[i,niter] < tol
                nlocked += 1
                vprintln("locked $nlocked")
            else
                # we lock in order, assuming that the lowest
                # eigenvectors converge first; might be tricky otherwise
                break
            end
        end

        display_progress && println("Iter $niter, converged $(nlocked)/$(n_conv_check), resid ", norm(resids[1:n_conv_check, niter]))

        if nlocked >= n_conv_check
            X .= new_X
            λ, full_X, residnorms, resids = final_residnorms(A, full_X, resids, niter)
            return λ, full_X, residnorms, resids
        end
        newly_locked = nlocked - prev_nlocked
        active = newly_locked+1:size(X,2) # newly active vectors

        if niter > 1
            ### compute P = Y*cP only for the newly active vectors
            Xn_indices = newly_locked+1:M-prev_nlocked
            # TODO understand this for a potential save of an
            # orthogonalization, see Hetmaniuk & Lehoucq, and Duersch et. al.
            # cP = copy(cX)
            # cP[Xn_indices,:] .= 0
            e = zeros(eltype(X), size(Y,2), M-prev_nlocked)
            for i in 1:length(Xn_indices)
                e[Xn_indices[i], i] = 1
            end
            cP = cX - e
            cP = cP[:, Xn_indices]
            # orthogonalize against all Xn (including newly locked)
            cP = ortho(cP, cX, cX, tol=ortho_tol)

            # Get new P
            new_P = Y*cP
            new_AP = AY*cP
            if B != I
                new_BP = BY*cP
            else
                new_BP = new_P
            end
        end

        # Update all X, even newly locked
        X .= new_X
        AX .= new_AX
        if B != I
            BX .= new_BX
        end

        # Sanity check
        for i = 1:size(X, 2)
            if abs(norm(view(X, :, i)) - 1) >= sqrt(eps(real(eltype(X))))
                error("LOBPCG is badly failing to keep the vectors normalized; this should never happen")
            end
        end

        # Restrict all arrays to active
        @views begin
            X = X[:, active]
            AX = AX[:, active]
            BX = BX[:, active]
            R = R[:, active]
            AR = AR[:, active]
            BR = BR[:, active]
            P = P[:, active]
            AP = AP[:, active]
            BP = BP[:, active]
            λs = λs[active]
        end

        # Update newly active P
        if niter > 1
            P .= new_P
            AP .= new_AP
            if B != I
                BP .= new_BP
            end
        end

        # Orthogonalize R wrt all X, newly active P
        if niter > 1
            Z = hcat(full_X, P)
            BZ = hcat(full_BX, BP)
        else
            Z = full_X
            BZ = full_BX
        end
        R = ortho(R, Z, BZ, tol=ortho_tol)

        if B != I
            # At this point R is orthogonalized but not B-orthogonalized.
            # We assume that B is relatively well-conditioned so that R is
            # close to be B-orthogonal. Therefore one step is OK, and B R
            # can be re-used
            mul!(BR, B, R)
            O = Hermitian(R'*BR)
            U = cholesky(O).U
            R = R/U
            BR = BR/U
            # condU = 1/LAPACK.trcon!('I', 'U', 'N', Array(U))
        else
            BR = R
        end

        niter = niter + 1
        niter <= maxiter || break
    end

    final_residnorms(A, full_X, resids, maxiter)
end
