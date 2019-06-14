using IterativeSolvers
import IterativeSolvers: lobpcg


"""
    lobpcg_itsolve(A, X0; prec=nothing, largest=false, kwargs...)

Call the `lobpcg` version from the `IterativeSolvers` package passing through most arguments
"""
function lobpcg_itsolve(A, X0; prec=nothing, largest=false, kwargs...)
    return lobpcg(A, largest, X0; P=prec, kwargs...)
end
