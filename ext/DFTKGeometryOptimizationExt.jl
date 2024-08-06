module DFTKGeometryOptimizationExt
using DFTK
import GeometryOptimization
GO = GeometryOptimization

function GO.minimize_energy!(system, calc::DFTKCalculator, solver;
                             autoadjust_calculator=true,
                             tol_energy=Inf*u"eV",
                             tol_force=1e-4u"eV/Å",
                             tol_virial=1e-6u"eV",
                             kwargs...)
    if autoadjust_calculator
        if :tol in keys(calc.ps.scf_kwargs) || :is_converged in keys(calc.ps.scf_kwargs)
            @warn("minimize_energy! automatically adjusts calculator SCF convergence parameters "
                  "unless keyword argument `autoadjust_calculator=false` is passed.")
        end

        # TODO I have no idea
        tol = min(austrip(tol_viral), austrip(tol_force), sqrt(austrip(tol_energy))) / 10
        scf_kwargs = merge(calc.ps.scf_kwargs, (; is_converged=ScfConvergenceDensity(tol), ))
        params = DFTKParameters(; calc.ps.model_kwargs, calc.ps.basis_kwargs, scf_kwargs)
        calc = DFTKCalculator(params, calc.st)
    end
    GO._minimize_energy!(system, calc, solver; tol_energy, tol_force, tol_virial, kwargs...)
end

end
