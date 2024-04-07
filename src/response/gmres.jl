using LinearMaps

function gmres(operators::Function, b::AbstractVector{T}; 
               x₀=nothing,
               maxiter=min(100, size(operators(0), 1)),
               restart=min(20, size(operators(0), 1)),
               tol=1e-6, s_guess=0.8, verbose=1, debug=false) where {T}
    # pass s/3m 1/\|r_tilde\| tol to operators()
    normb = norm(b)
    b = b ./ normb
    tol = tol / normb
    if tol < 0.5*eps(T)
        #throw a warning
        @warn "The effective tolerance is smaller than half of the machine epsilon, the requested accuracy may not be achieved."
    end

    n_size = length(b)

    # counters
    numrestarts::Int64 = 0
    restart_inds = Int64[]
    MvTime = Float64[]

    # initialization: Arnoldi basis and Hessenberg matrix
    V = zeros(n_size, restart + 1)
    H = zeros(restart + 1, restart)

    # residual history
    residuals = zeros(maxiter)
    denominatorvec = zeros(restart)
    denominator2 = 1.0
    # record min singular values
    minsvdvals = zeros(maxiter)

    # solution history
    if debug
        X = zeros(n_size, maxiter)
    end

    # if x0 is zero, set r0 = b
    x0iszero = isnothing(x₀)
    if x0iszero
        β₀ = 1.0
        V[:, 1] = b #/ β₀
        tol_new = tol / 2
        lm = s_guess / restart / 2 * tol
    else
        # assert x₀ is a vector of the right size
        @assert length(x₀) == n_size
        x₀ ./= normb
        x0iszero = false
        tol_new = tol / 3
        A0_tol = tol / 3
        lm = s_guess / restart / 3 * tol
        push!(MvTime, @elapsed r₀ = operators(A0_tol) * x₀)
        r₀ = b - r₀
        β₀ = norm(r₀)
        V[:, 1] = r₀ / β₀
    end

    Ai_tol = lm / β₀ 
    i = 0 # iteration counter, set to 0 when restarted
    for j = 1:maxiter
        i += 1

        # Arnoldi iteration using MGS
        push!(MvTime, @elapsed w = operators(Ai_tol) * V[:, i])
        for j = 1:i
            H[j, i] = dot(V[:, j], w)
            w .-= H[j, i] * V[:, j]
        end
        H[i+1, i] = norm(w)
        V[:, i+1] = w / H[i+1, i]

        # compute the min singular value of the upper Hessenberg matrix
        minsvdvals[j] = svdvals(H[1:i+1, 1:i])[end]
    
        # update residual history
        if i == 1
            denominatorvec[1] = conj(-H[1, 1] / H[2, 1])
        else
            denominatorvec[i] = conj(-(H[1, i] + dot(denominatorvec[1:i-1], H[2:i, i])) / H[i+1, i])
        end
        denominator2 += abs2(denominatorvec[i])
        residuals[j] = β₀ / sqrt(denominator2)
        Ai_tol = lm / residuals[j]

        (verbose > 0) && println("restart = ", numrestarts + 1, ", iteration = ", i, ", res = ", residuals[j]*normb, "\n")

        # when converged without updating the s_guess, restart once
        converged = residuals[j] < tol_new
        happy = converged && ((numrestarts > 0) || (s_guess <= minsvdvals[j]))
        needrestart = (i == restart) || (converged && !happy) # equal to converged && (numrestarts == 0) && (s_guess > minsvdvals[j])
        reachmaxiter = (j == maxiter)
        exitloop = (happy || reachmaxiter)

        solvels = (exitloop || needrestart || debug)
        if solvels
            # solve the Hessenberg least squares problem
            rhs = (UpperTriangular(H[2:i+1, 1:i]) \ denominatorvec[1:i]) * (-β₀ / denominator2)
            if x0iszero
                xᵢ = V[:, 1:i] * rhs
            else
                xᵢ = x₀ + V[:, 1:i] * rhs
            end

            debug && (X[:, j] = xᵢ)
            
            if exitloop
                if debug
                    return (; x=X[:, 1:j].*normb, residuals=residuals[1:j].*normb, MvTime=MvTime, restart_inds=restart_inds, minsvdvals=minsvdvals[1:j])
                else
                    return (; x=xᵢ.*normb, residuals=residuals[1:j].*normb, MvTime=MvTime, restart_inds=restart_inds)
                end
            end

            if needrestart
                i = 0
                x0iszero = false
                # whenever restart, update the guess of the smallest singular value
                min_s = minimum(minsvdvals[1:j])
                if s_guess > min_s
                    verbose > 0 && println("current s_guess = ", s_guess, "> minsvdval = ", min_s, ", restart with s_guess = minsvdval")
                else
                    verbose > 0 && println("current s_guess = ", s_guess, "≤ minsvdval = ", min_s, ", restart with s_guess = minsvdval")
                end
                s_guess = min_s
                lm = s_guess / restart / 3 * tol
                tol_new = tol / 3
                A0_tol = tol / 3
                
                x₀ = xᵢ
                push!(MvTime, @elapsed r₀ = operators(A0_tol) * x₀)
                r₀ = b - r₀
                β₀ = norm(r₀)
                V[:, 1] = r₀ / β₀
                numrestarts += 1
                denominatorvec = zeros(maxiter)
                denominator2 = 1.0
                push!(restart_inds, j)
                (verbose > -1) && println("restarting: ", "iteration = ", j, ", r₀ = ", β₀, "\n")
                Ai_tol = lm / β₀ 
            end
        end
    end
end


function gmres(operator::LinearMap{T}, b::AbstractVector{T}; 
               x₀ = nothing,
               maxiter=min(100, size(operator, 1)),
               restart=min(20, size(operator, 1)),
               tol=1e-6, verbose=0, debug=false) where {T}

    n_size = length(b)

    # counters
    numrestarts::Int64 = 0
    restart_inds = Int64[]
    MvTime = Float64[]

    # initialization: Arnoldi basis and Hessenberg matrix
    V = zeros(n_size, restart + 1)
    H = zeros(restart + 1, restart)

    # residual history
    residuals = zeros(maxiter)
    denominatorvec = zeros(restart)
    denominator2 = 1.0

    # solution history
    if debug
        X = zeros(n_size, maxiter)
    end

    # if x0 is zero, set r0 = b
    if isnothing(x₀)
        β₀ = norm(b)
        V[:, 1] = b / β₀
    else
        # assert x₀ is a vector of the right size
        @assert length(x₀) == n_size
        push!(MvTime, @elapsed r₀ = operator * x₀)
        r₀ = b - r₀
        β₀ = norm(r₀)
        V[:, 1] = r₀ / β₀
    end

    i = 0 # iteration counter, set to 0 when restarted
    for j = 1:maxiter
        i += 1
        # Arnoldi iteration using MGS
        push!(MvTime, @elapsed w = operator * V[:, i])
        for j = 1:i
            H[j, i] = dot(V[:, j], w)
            w .-= H[j, i] * V[:, j]
        end
        H[i+1, i] = norm(w)
        V[:, i+1] = w / H[i+1, i]

        # update residual history
        if i == 1
            denominatorvec[1] = conj(-H[1, 1] / H[2, 1])
        else
            denominatorvec[i] = conj(-(H[1, i] + dot(denominatorvec[1:i-1], H[2:i, i])) / H[i+1, i])
        end
        denominator2 += abs2(denominatorvec[i])
        residuals[j] = β₀ / sqrt(denominator2)
        
        (verbose > 0) && println("restart = ", numrestarts + 1, ", iteration = ", i, ", res = ", residuals[j], "\n")

        happy = residuals[j] < tol
        needrestart = (i == restart)
        reachmaxiter = (j == maxiter)
        exitloop = (happy || reachmaxiter)

        solvels = (exitloop || needrestart || debug)
        if solvels
            # solve the Hessenberg least squares problem
            rhs = (UpperTriangular(H[2:i+1, 1:i]) \ denominatorvec[1:i]) * (-β₀ / denominator2)
            if isnothing(x₀)
                xᵢ = V[:, 1:i] * rhs
            else
                xᵢ = x₀ + V[:, 1:i] * rhs
            end

            debug && (X[:, j] = xᵢ)
            
            if exitloop
                if debug
                    return (; x=X[:, 1:j], residuals=residuals[1:j], MvTime=MvTime, restart_inds=restart_inds)
                else
                    return (; x=xᵢ, residuals=residuals[1:j], MvTime=MvTime, restart_inds=restart_inds)
                end
            end

            if needrestart
                i = 0
                x₀ = xᵢ
                push!(MvTime, @elapsed r₀ = operator * x₀)
                r₀ = b - r₀
                β₀ = norm(r₀)
                V[:, 1] = r₀ / β₀
                numrestarts += 1
                denominatorvec = zeros(maxiter)
                denominator2 = 1.0
                push!(restart_inds, j)
                (verbose > 0) && println("restarting: ", "iteration = ", j, ", r₀ = ", β₀, "\n")
            end
        end
    end
end