using DFTK
using LinearAlgebra
using PseudoPotentialData
using AtomsBuilder

# We take very (very) crude parameters
pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
model = model_DFT(bulk(:Si); functionals=LDA(), pseudopotentials)
basis = PlaneWaveBasis(model; Ecut=5, kgrid=[1, 1, 1]);

function my_fp_solver(f, x0, info0; maxiter, damping=0.7)
    x = x0
    info = info0
    for n = 1:maxiter
        fx, info = f(x, info)
        if info.converged || info.timedout
            break
        end
        x = x + damping * (fx - x)
    end
    (; fixpoint=x, info)
end;

scfres = self_consistent_field(basis;
                               tol=1e-4,
                               #solver=my_fp_solver,
                               #mixing=MyMixing(),
                               maxiter=100);  # to the corresponding keywords of my_fp_solver
