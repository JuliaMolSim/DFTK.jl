# # Custom solvers
# In this example, we show how to define custom solvers. Our system
# will again be silicon, because we are not very imaginative
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

# Finally we also define our custom mixing scheme. It will be a mixture
# of simple mixing (for the first 2 steps) and than default to Kerker mixing.
# In the mixing interface `δF` is ``(ρ_\text{out} - ρ_\text{in})``, i.e.
# the difference in density between two subsequent SCF steps and the `mix`
# function returns ``δρ``, which is added to ``ρ_\text{in}`` to yield ``ρ_\text{next}``,
# the density for the next SCF step.
struct MyMixing
    α  # Damping parameter
end
MyMixing() = MyMixing(0.7)

function DFTK.mix(mixing::MyMixing, basis, δF; n_iter, kwargs...)
    if n_iter <= 2
        ## Just do simple mixing
        mixing.α * δF
    else
        ## Use the KerkerMixing from DFTK
        DFTK.mix(KerkerMixing(α=mixing.α), basis, δF; kwargs...)
    end
end

# That's it! Now we just run the SCF with these solvers
scfres = self_consistent_field(basis;
                               tol=1e-8,
                               solver=my_fp_solver,
                               eigensolver=my_eig_solver,
                               mixing=MyMixing());
# Note that the default convergence criterion is on the difference of
# energy from one step to the other; when this gets below `tol`, the
# "driver" `self_consistent_field` artificially makes the fixpoint
# solver think it's converged by forcing `f(x) = x`. You can customize
# this with the `is_converged` keyword argument to
# `self_consistent_field`.
