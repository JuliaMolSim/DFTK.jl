# these provide fixed-point solvers that can be passed to scf()

# the fp_solver function must accept being called like fp_solver(f, x0, tol,
# maxiter), where f(x) is the fixed-point map. It must return an
# object supporting res.sol and res.converged

function scf_nlsolve_solver(m)
    function fp_solver(f, x0, tol, max_iter)
        res = nlsolve(x -> f(x) - x, x0, method=:anderson, m=m, xtol=tol,
                      ftol=0.0, show_trace=true, iterations=max_iter)
        (sol=res.zero, converged=converged(res))
    end
    fp_solver
end
function scf_damping_solver(β)
    function fp_solver(f, x0, tol, max_iter)
        converged = false
        x = copy(x0)
        for i in 1:max_iter
            x_new = f(x)

            # TODO Print statements should not be here
            ndiff = norm(x_new - x)
            @printf "%4d %18.8g\n" i ndiff

            if 20 * ndiff < tol
                x = x_new
                converged = true
                break
            end

            x = @. β * x_new + (1 - β) * x
        end
        (sol=x, converged=converged)
    end
    fp_solver
end



## Basic versions of anderson mixing and CROP algorithm

function anderson(g,x0,m::Int,niter::Int,eps::Real,warming=0)
    @assert length(size(x0)) == 1 #1D array input
    N = size(x0,1)
    T = eltype(x0)
    #xs: ring buffer storing the iterates, from newest to oldest
    xs = zeros(T,N,m+1)
    fs = zeros(T,N,m+1)
    xs[:,1] = x0
    errs = zeros(niter)
    err = Inf

    for n = 1:niter
        fs[:,1] = g(xs[:,1])-xs[:,1]
        err = norm(fs[:,1])
        errs[n] = err
        println("$n $err")
        if(err < eps)
            break
        end
        m_eff = min(n-1,m)
        new_x = xs[:,1]+fs[:,1]
        if m_eff > 0 && n > warming
            mat = fs[:,2:m_eff+1] .- fs[:,1]
            alphas = -mat \ fs[:,1]
            # alphas = -(mat'*mat) \ (mat'* (gs[:,1] - xs[:,1]))
            for i = 1:m_eff
                new_x .+= alphas[i].*(xs[:,i+1] + fs[:,i+1] - xs[:,1] - fs[:,1])
            end
        end

        xs = circshift(xs,(0,1))
        fs = circshift(fs,(0,1))
        xs[:,1] = new_x
    end
    (sol=xs[:,1], converged=err < eps)
end

# CROP iterates maintain xn and fn (/!\ fn != f(xn)).
# xtn+1 = xn + fn
# ftn+1 = f(xtn+1)
# Determine αi from min ftn+1 + sum αi(fi - ftn+1)
# fn+1 = ftn+1 + sum αi(fi - ftn+1)
# xn+1 = xtn+1 + sum αi(xi - xtn+1)

function CROP(g,x0,m::Int,niter::Int,eps::Real,warming=0)
    @assert length(size(x0)) == 1 #1D array input
    N = size(x0,1)
    T = eltype(x0)
    #xs: ring buffer storing the iterates, from newest to oldest
    xs = zeros(T,N,m+1)
    fs = zeros(T,N,m+1)
    xs[:,1] = x0
    fs[:,1] = g(x0) - x0
    errs = zeros(niter)
    err = Inf

    for n = 1:niter
        # println(xs[1:4,1])
        xtnp1 = xs[:,1] + fs[:,1]
        ftnp1 = g(xtnp1)-xtnp1
        err = norm(ftnp1)
        errs[n] = err
        println("$n $err")
        if(err < eps)
            break
        end
        m_eff = min(n,m)
        if m_eff > 0 && n > warming
            mat = fs[:,1:m_eff] .- ftnp1
            alphas = -mat \ ftnp1
            bak_xtnp1 = copy(xtnp1)
            bak_ftnp1 = copy(ftnp1)
            for i = 1:m_eff
                xtnp1 .+= alphas[i].*(xs[:,i] .- bak_xtnp1)
                ftnp1 .+= alphas[i].*(fs[:,i] .- bak_ftnp1)
            end
            # println(norm(ftnp1 - (bak_ftnp1 + mat*alphas)))
        end

        xs = circshift(xs,(0,1))
        fs = circshift(fs,(0,1))
        xs[:,1] = xtnp1
        fs[:,1] = ftnp1
        # fs[:,1] = g(xs[:,1])-xs[:,1]
    end
    (sol=xs[:,1], converged=err < eps)
end
scf_anderson_solver(m) = (f, x0, tol, max_iter) -> anderson(f,x0,m,max_iter,tol)
scf_CROP_solver(m) = (f, x0, tol, max_iter) -> CROP(f,x0,m,max_iter,tol)
