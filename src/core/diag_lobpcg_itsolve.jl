import IterativeSolvers

"""
    lobpcg_itsolve(A, X0; prec=nothing, largest=false, kwargs...)

Call the `lobpcg` version from the `IterativeSolvers` package passing through most arguments
"""
function diag_lobpcg_itsolve(;kwargs...)
    function inner(A, X0; largest=false, prec=nothing, kwargs...)
        IterativeSolvers.lobpcg(A, largest, X0; P=prec, kwargs...)
    end
    construct_diag(inner; kwargs...)
end
