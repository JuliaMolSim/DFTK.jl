module DFTKGeometryOptimizationExt
using DFTK
using Unitful
using UnitfulAtomic
import GeometryOptimization
GO = GeometryOptimization

# For DFTK use LBFGS as default algorithm
# TODO Benchmark this heuristics
function GO.setup_solver(system, calc::DFTKCalculator, ::GO.Autoselect; kwargs...)
    GO.setup_solver(system, calc, GO.OptimLBFGS(); kwargs...)
end

function GO.minimize_energy!(system, calc::DFTKCalculator, solver;
                             autoadjust_calculator=true,
                             tol_energy=Inf*u"eV",
                             tol_forces=1e-4u"eV/Å",
                             tol_virial=1e-6u"eV",
                             verbosity::Integer=0,
                             # Verbosity is 0 (silent) unless we are on master
                             callback=GO.GeoOptDefaultCallback(mpi_master() * verbosity),
                             kwargs...)
    if autoadjust_calculator
        has_modified_parameter = any(haskey(calc.params.scf_kwargs, key)
                                     for key in (:tol, :is_converged, :callback))
        has_identity_callback = (haskey(calc.params.scf_kwargs, :callback)
                                 && calc.params.scf_kwargs.callback == identity)
        if has_modified_parameter && !has_identity_callback
            @warn("minimize_energy! automatically adjusts some SCF convergence and " *
                  "printing parameters unless the keyword argument " *
                  "`autoadjust_calculator=false` is passed.")
        end

        # TODO Hardly tested, whether these heuristics are reasonable
        tol = min(austrip(tol_virial), austrip(tol_forces), sqrt(austrip(tol_energy))) / 10
        scf_kwargs = merge(calc.params.scf_kwargs,
                           (; miniter=2, is_converged=ScfConvergenceDensity(tol), ))
        if verbosity > 1
            scf_kwargs = merge(scf_kwargs, (; callback=ScfDefaultCallback()))
        end

        params = DFTK.DFTKParameters(; calc.params.model_kwargs,
                                       calc.params.basis_kwargs,
                                       scf_kwargs)
        calc = DFTKCalculator(params, calc.st; enforce_convergence=false)
    end

    callback_dftk = function (optim_state, geoopt_state)
        if !geoopt_state.calc_state.converged  # SCF not converged
            return true  # Halt geometry optimisation
        end
        return callback(optim_state, geoopt_state)
    end
    GO._minimize_energy!(system, calc, solver;
                         tol_energy, tol_forces, tol_virial,
                         verbosity, callback=callback_dftk, kwargs...)
end

end
