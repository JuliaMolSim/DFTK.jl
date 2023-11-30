# The Hessian of P -> E(P) (E being the energy) is Ω+K, where Ω and K are
# defined below (cf. [1] for more details).
#
# In particular, by linearizing the Kohn-Sham equations, we have
#
# δP = -(Ω+K)⁻¹ δH
#
# which can be solved either directly (solve_ΩplusK) or by a splitting method
# (solve_ΩplusK_split), the latter being preferable as it is well defined for
# both metals and insulators. Solving this equation is necessary to compute
# response properties as well as AD chain rules (Ω+K being self-adjoint, it can
# be used for both forward and reverse mode).
#
# [1] Eric Cancès, Gaspard Kemlin, Antoine Levitt. Convergence analysis of
#     direct minimization and self-consistent iterations.
#     SIAM Journal on Matrix Analysis and Applications
#     https://doi.org/10.1137/20M1332864
#
# TODO find better names for solve_ΩplusK and solve_ΩplusK_split
#

"""
    apply_Ω(δψ, ψ, H::Hamiltonian, Λ)

Compute the application of Ω defined at ψ to δψ. H is the Hamiltonian computed
from ψ and Λ is the set of Rayleigh coefficients ψk' * Hk * ψk at each k-point.
"""
@timing function apply_Ω(δψ, ψ, H::Hamiltonian, Λ)
    δψ = proj_tangent(δψ, ψ)
    Ωδψ = [H.blocks[ik] * δψk - δψk * Λ[ik] for (ik, δψk) in enumerate(δψ)]
    proj_tangent!(Ωδψ, ψ)
end

"""
    apply_K(basis::PlaneWaveBasis, δψ, ψ, ρ, occupation)

Compute the application of K defined at ψ to δψ. ρ is the density issued from ψ.
δψ also generates a δρ, computed with `compute_δρ`.
"""
@views @timing function apply_K(basis::PlaneWaveBasis, δψ, ψ, ρ, occupation)
    δψ = proj_tangent(δψ, ψ)
    δρ = compute_δρ(basis, ψ, δψ, occupation)
    δV = apply_kernel(basis, δρ; ρ)

    Kδψ = map(enumerate(ψ)) do (ik, ψk)
        kpt = basis.kpoints[ik]
        δVψk = similar(ψk)

        for n = 1:size(ψk, 2)
            ψnk_real = ifft(basis, kpt, ψk[:, n])
            δVψnk_real = δV[:, :, :, kpt.spin] .* ψnk_real
            δVψk[:, n] = fft(basis, kpt, δVψnk_real)
        end
        δVψk
    end
    # ensure projection onto the tangent space
    proj_tangent!(Kδψ, ψ)
end

"""
    solve_ΩplusK(basis::PlaneWaveBasis{T}, ψ, res, occupation;
                 tol=1e-10, verbose=false) where {T}

Return δψ where (Ω+K) δψ = rhs
"""
@timing function solve_ΩplusK(basis::PlaneWaveBasis{T}, ψ, rhs, occupation;
                              callback=identity, tol=1e-10) where {T}
    filled_occ = filled_occupation(basis.model)
    # for now, all orbitals have to be fully occupied -> need to strip them beforehand
    @assert all(all(occ_k .== filled_occ) for occ_k in occupation)

    # To mpi-parallelise we have to deal with the fact that the linear algebra
    # in the CG (dot products, norms) couples k-Points. Maybe take a look at
    # the PencilArrays.jl package to get this done automatically.
    @assert mpi_nprocs() == 1  # Distributed implementation not yet available

    # compute quantites at the point which define the tangent space
    ρ = compute_density(basis, ψ, occupation)
    H = energy_hamiltonian(basis, ψ, occupation; ρ).ham

    pack(ψ) = reinterpret_real(pack_ψ(ψ))
    unpack(x) = unpack_ψ(reinterpret_complex(x), size.(ψ))
    unsafe_unpack(x) = unsafe_unpack_ψ(reinterpret_complex(x), size.(ψ))

    # project rhs on the tangent space before starting
    proj_tangent!(rhs, ψ)
    rhs_pack = pack(rhs)

    # preconditioner
    Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    for ik = 1:length(Pks)
        precondprep!(Pks[ik], ψ[ik])
    end
    function f_ldiv!(x, y)
        δψ = unpack(y)
        proj_tangent!(δψ, ψ)
        Pδψ = [ Pks[ik] \ δψk for (ik, δψk) in enumerate(δψ)]
        proj_tangent!(Pδψ, ψ)
        x .= pack(Pδψ)
    end

    # Rayleigh-coefficients
    Λ = [ψk'Hψk for (ψk, Hψk) in zip(ψ, H * ψ)]

    # mapping of the linear system on the tangent space
    function ΩpK(x)
        δψ = unsafe_unpack(x)
        Kδψ = apply_K(basis, δψ, ψ, ρ, occupation)
        Ωδψ = apply_Ω(δψ, ψ, H, Λ)
        pack(Ωδψ + Kδψ)
    end
    J = LinearMap{T}(ΩpK, size(rhs_pack, 1))

    # solve (Ω+K) δψ = rhs on the tangent space with CG
    function proj(x)
        δψ = unpack(x)
        proj_tangent!(δψ, ψ)
        pack(δψ)
    end
    res = cg(J, rhs_pack; precon=FunctionPreconditioner(f_ldiv!), proj, tol,
             callback)
    (; δψ=unpack(res.x), res.converged, res.tol, res.residual_norm,
     res.n_iter)
end


"""
Solve the problem `(Ω+K) δψ = rhs` using a split algorithm, where `rhs` is typically
`-δHextψ` (the negative matvec of an external perturbation with the SCF orbitals `ψ`) and
`δψ` is the corresponding total variation in the orbitals `ψ`. Additionally returns:
    - `δρ`:  Total variation in density)
    - `δHψ`: Total variation in Hamiltonian applied to orbitals
    - `δeigenvalues`: Total variation in eigenvalues
    - `δVind`: Change in potential induced by `δρ` (the term needed on top of `δHextψ`
      to get `δHψ`).
"""
@timing function solve_ΩplusK_split(ham::Hamiltonian, ρ::AbstractArray{T}, ψ, occupation, εF,
                                    eigenvalues, rhs; tol=1e-8, tol_sternheimer=tol/10,
                                    verbose=false, occupation_threshold, q=zero(Vec3{real(T)}),
                                    kwargs...) where {T}
    # Using χ04P = -Ω^-1, E extension operator (2P->4P) and R restriction operator:
    # (Ω+K)^-1 =  Ω^-1 ( 1 -   K   (1 + Ω^-1 K  )^-1    Ω^-1  )
    #          = -χ04P ( 1 -   K   (1 - χ04P K  )^-1   (-χ04P))
    #          =  χ04P (-1 + E K2P (1 - χ02P K2P)^-1 R (-χ04P))
    # where χ02P = R χ04P E and K2P = R K E
    basis = ham.basis
    @assert size(rhs[1]) == size(ψ[1])  # Assume the same number of bands in ψ and rhs

    # compute δρ0 (ignoring interactions)
    δψ0, δoccupation0 = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, -rhs;
                                    tol=tol_sternheimer, occupation_threshold, q,
                                    kwargs...)  # = -χ04P * rhs
    δρ0 = compute_δρ(basis, ψ, δψ0, occupation, δoccupation0; occupation_threshold, q)

    # compute total δρ
    pack(δρ)   = vec(δρ)
    unpack(δρ) = reshape(δρ, size(ρ))
    function eps_fun(δρ)
        δρ = unpack(δρ)
        δV = apply_kernel(basis, δρ; ρ, q)
        # TODO
        # Would be nice to play with abstol / reltol etc. to avoid over-solving
        # for the initial GMRES steps.
        χ0δV = apply_χ0(ham, ψ, occupation, εF, eigenvalues, δV;
                        occupation_threshold, tol=tol_sternheimer, q, kwargs...)
        pack(δρ - χ0δV)
    end
    J = LinearMap{T}(eps_fun, prod(size(δρ0)))
    δρ, history = gmres(J, pack(δρ0); reltol=0, abstol=tol, verbose, log=true)
    δρ = unpack(δρ)

    # Compute total change in Hamiltonian applied to ψ
    δVind = apply_kernel(basis, δρ; ρ, q)  # Change in potential induced by δρ
    # For phonon calculations, assemble
    #   δHψ_k = δV_{q} · ψ_{k-q}.
    δHψ = multiply_ψ_by_blochwave(basis, ψ, δVind, q) - rhs

    # Compute total change in eigenvalues
    δeigenvalues = map(ψ, δHψ) do ψk, δHψk
        map(eachcol(ψk), eachcol(δHψk)) do ψnk, δHψnk
            real(dot(ψnk, δHψnk))  # δε_{nk} = <ψnk | δH | ψnk>
        end
    end

    (; δψ, δoccupation, δεF) = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, δHψ;
                                           occupation_threshold, tol=tol_sternheimer, q,
                                           kwargs...)

    (; δψ, δρ, δHψ, δVind, δeigenvalues, δoccupation, δεF, history)
end

function solve_ΩplusK_split(scfres::NamedTuple, rhs; kwargs...)
    solve_ΩplusK_split(scfres.ham, scfres.ρ, scfres.ψ, scfres.occupation,
                       scfres.εF, scfres.eigenvalues, rhs;
                       scfres.occupation_threshold, kwargs...)
end
