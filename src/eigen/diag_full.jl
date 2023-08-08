function diag_full(A, X0; kwargs...)
    Neig = size(X0, 2)
    Afull = Hermitian(Array(A))
    E = eigen(Afull)
    X = E.vectors[:, 1:Neig]
    λ = E.values[1:Neig]
    (; λ, X,
     residual_norms=zeros(Neig),
     n_iter=0,
     converged=true,
     n_matvec=0)
end
