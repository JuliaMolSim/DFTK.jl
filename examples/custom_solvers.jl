# # [Custom solvers](@id custom-solvers)
# In this example, we show how to define custom solvers. Our system
# will again be silicon, because we are not very imaginative
using DFTK, LinearAlgebra

a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si, Si]
positions =  [ones(3)/8, -ones(3)/8]

## We take very (very) crude parameters
model = model_LDA(lattice, atoms, positions)
basis = PlaneWaveBasis(model; Ecut=5, kgrid=[1, 1, 1]);

# We define our custom fix-point solver: simply a damped fixed-point
function my_fp_solver(f, x0, info0; maxiter)
    mixing_factor = .7
    x = x0
    info = info0
    for n = 1:maxiter
        fx, info = f(x, info)
        if info.converged || info.timedout
            break
        end
        x = x + mixing_factor * (fx - x)
    end
    (; fixpoint=x, info)
end;
# Note that the fixpoint map `f` operates on an auxiliary variable `info` for
# state bookkeeping. Early termination criteria are flagged from inside
# the function `f` using boolean flags `info.converged` and `info.timedout`.
# For control over these criteria, see the `is_converged` and `maxtime`
# keyword arguments of `self_consistent_field`.

# Our eigenvalue solver just forms the dense matrix and diagonalizes
# it explicitly (this only works for very small systems)
function my_eig_solver(A, X0; maxiter, tol, kwargs...)
    n = size(X0, 2)
    A = Array(A)
    E = eigen(A)
    λ = E.values[1:n]
    X = E.vectors[:, 1:n]
    (; λ, X, residual_norms=[], n_iter=0, converged=true, n_matvec=0)
end;

# Finally we also define our custom mixing scheme. It will be a mixture
# of simple mixing (for the first 2 steps) and than default to Kerker mixing.
# In the mixing interface `δF` is ``(ρ_\text{out} - ρ_\text{in})``, i.e.
# the difference in density between two subsequent SCF steps and the `mix`
# function returns ``δρ``, which is added to ``ρ_\text{in}`` to yield ``ρ_\text{next}``,
# the density for the next SCF step.
struct MyMixing
    n_simple  # Number of iterations for simple mixing
end
MyMixing() = MyMixing(2)

function DFTK.mix_density(mixing::MyMixing, basis, δF; n_iter, kwargs...)
    if n_iter <= mixing.n_simple
        return δF  # Simple mixing -> Do not modify update at all
    else
        ## Use the default KerkerMixing from DFTK
        DFTK.mix_density(KerkerMixing(), basis, δF; kwargs...)
    end
end

# That's it! Now we just run the SCF with these solvers
scfres = self_consistent_field(basis;
                               tol=1e-4,
                               solver=my_fp_solver,
                               eigensolver=my_eig_solver,
                               mixing=MyMixing());
# Note that the default convergence criterion is the difference in
# density. When this gets below `tol`, the fixed-point solver terminates.
# You can also customize this with the `is_converged` keyword argument to
# `self_consistent_field`, as shown below.

# ## Customizing the convergence criterion
# Here is an example of a defining a custom convergence criterion and specifying
# it using the `is_converged` callback keyword to `self_consistent_field`.

function my_convergence_criterion(info)
    tol = 1e-10
    length(info.history_Etot) < 2 && return false
    ΔE = (info.history_Etot[end-1] - info.history_Etot[end])
    ΔE < tol
end

scfres2 = self_consistent_field(basis;
                                solver=my_fp_solver,
                                is_converged=my_convergence_criterion,
                                eigensolver=my_eig_solver,
                                mixing=MyMixing());
