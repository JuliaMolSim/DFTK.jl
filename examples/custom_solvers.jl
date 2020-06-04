# # Custom solvers
# In this example, we show how to define custom solvers. Our system will again be silicon, because we are not very imaginative
using DFTK, LinearAlgebra

a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

## We take very (very) crude parameters
model = model_LDA(lattice, atoms)
kgrid = [1, 1, 1]
Ecut = 5
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid);

# We define our custom fix-point solver: simply a damped fixed-point
function my_fp_solver(f, x0, max_iter; tol)
    mixing_factor = .7
    x = x0
    fx = f(x)
    for n = 1:max_iter
        inc = fx - x
        ## Note that the convergence criterion might be overwritten by the SCF solver
        if norm(inc) < tol
            break
        end
        x = x + mixing_factor * inc
        fx = f(x)
    end
    (fixpoint=x, converged=norm(fx-x) < tol)
end;

# Our eigenvalue solver just forms the dense matrix and diagonalizes
# it explicitly (this only works for very small systems)
function my_eig_solver(A, X0; maxiter, tol, kwargs...)
    n = size(X0, 2)
    A = Array(A)
    E = eigen(A)
    λ = E.values[1:n]
    X = E.vectors[:, 1:n]
    (λ=λ, X=X, residual_norms=[], iterations=0, converged=true, n_matvec=0)
end;

# That's it! Now we just run the SCF with these solvers
scfres = self_consistent_field(basis, tol=1e-8, solver=my_fp_solver, eigensolver=my_eig_solver);
