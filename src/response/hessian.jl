using KrylovKit

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
@views @timing function apply_K(basis::PlaneWaveBasis{T}, δψ, ψ, ρ, occupation) where {T}
    # ~45% of apply_K is spent computing ifft(ψ) twice: once in compute_δρ and once again below.
    # By caching the result, we could compute it only once for a single application of K,
    # or even across many applications when using solve_ΩplusK.
    # But we don't because the memory requirements would be too high (typically an order of magnitude higher than ψ).

    δψ = proj_tangent(δψ, ψ)
    δρ = compute_δρ(basis, ψ, δψ, occupation)
    δV = apply_kernel(basis, δρ; ρ)

    ψnk_real = similar(G_vectors(basis), promote_type(T, eltype(ψ[1])))
    Kδψ = map(enumerate(ψ)) do (ik, ψk)
        kpt = basis.kpoints[ik]
        δVψk = similar(ψk)

        for n = 1:size(ψk, 2)
            ifft!(ψnk_real, basis, kpt, ψk[:, n])
            ψnk_real .*= δV[:, :, :, kpt.spin]
            fft!(δVψk[:, n], basis, kpt, ψnk_real)
        end
        δVψk
    end
    # ensure projection onto the tangent space
    proj_tangent!(Kδψ, ψ)
end

"""
Default callback function for `solve_ΩplusK`,
which prints a convergence table.
"""
struct ResponseCallback
    prev_time::Ref{UInt64}
end
function ResponseCallback()
    ResponseCallback(Ref(zero(UInt64)))
end
function (cb::ResponseCallback)(info)
    mpi_master() || return info  # Only print on master

    if info.stage == :finalize
        info.converged || @warn "solve_ΩplusK not converged."
        return info
    end

    if info.n_iter == 0
        cb.prev_time[] = time_ns()
        @printf "n     log10(Residual norm)   Δtime \n"
        @printf "---   --------------------   ------\n"
        return info
    end

    current_time = time_ns()
    runtime_ns = current_time - cb.prev_time[]
    cb.prev_time[] = current_time

    resnorm = @sprintf "%20.2f" log10(info.residual_norm)
    time = @sprintf "% 6s" TimerOutputs.prettytime(runtime_ns)
    @printf "% 3d   %s   %s\n" info.n_iter resnorm time
    flush(stdout)
    info
end

"""
Solve density-functional perturbation theory problem,
that is return δψ where (Ω+K) δψ = rhs.
"""
@timing function solve_ΩplusK(basis::PlaneWaveBasis{T}, ψ, rhs, occupation;
                              callback=ResponseCallback(), tol=1e-10) where {T}
    # for now, all orbitals have to be fully occupied -> need to strip them beforehand
    check_full_occupation(basis, occupation)

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
             callback, comm=basis.comm_kpts)
    (; δψ=unpack(res.x), res.converged, res.tol, res.residual_norm,
     res.n_iter)
end


"""
Solve the problem `(Ω+K) δψ = rhs` (density-functional perturbation theory)
using a split algorithm, where `rhs` is typically
`-δHextψ` (the negative matvec of an external perturbation with the SCF orbitals `ψ`) and
`δψ` is the corresponding total variation in the orbitals `ψ`. Additionally returns:
    - `δρ`:  Total variation in density
    - `δHψ`: Total variation in Hamiltonian applied to orbitals
    - `δeigenvalues`: Total variation in eigenvalues
    - `δVind`: Change in potential induced by `δρ` (the term needed on top of `δHextψ`
      to get `δHψ`).

Input parameters:
- `tol`: Desired tolerance in the density variation and orbital variation
- `bandtolalg`: Algorithm for adaptive selection of Sternheimer tolerances,
  see [arxiv 2505.02319](https://arxiv.org/pdf/2505.02319) for more details.
- `mixing`: Mixing to use to precondition GMRES
"""
@timing function solve_ΩplusK_split(ham::Hamiltonian, ρ::AbstractArray{T}, ψ, occupation, εF,
                                    eigenvalues, rhs;
                                    tol=1e-8, verbose=false,
                                    mixing=LdosMixing(; adjust_temperature=UseScfTemperature()),
                                    factor_initial=1/10, factor_final=1/10,
                                    occupation_threshold,
                                    bandtolalg=BandtolBalanced(ham.basis, ψ, occupation,
                                                               eigenvalues; occupation_threshold),
                                    q=zero(Vec3{real(T)}), kwargs...) where {T}
    # Using χ04P = -Ω^-1, E extension operator (2P->4P) and R restriction operator:
    # (Ω+K)^-1 =  Ω^-1 ( 1 -   K   (1 + Ω^-1 K  )^-1    Ω^-1  )
    #          = -χ04P ( 1 -   K   (1 - χ04P K  )^-1   (-χ04P))
    #          =  χ04P (-1 + E K2P (1 - χ02P K2P)^-1 R (-χ04P))
    # where χ02P = R χ04P E and K2P = R K E
    basis = ham.basis
    @assert size(rhs[1]) == size(ψ[1])  # Assume the same number of bands in ψ and rhs

    # TODO Better initial guess handling. Especially between the last iteration of the GMRES
    #      and the concluding Sternheimer solve we should be able to benefit from passing
    #      around the orbitals

    # TODO Use tol_density=tol/10 to make sure that the density is very accurate.
    #      This is likely overdoing it and we should investigate if a smaller
    #      value also does the trick.

    # compute δρ0 (ignoring interactions)
    bandtol0 = determine_band_tolerances(bandtolalg, tol * factor_initial)
    res0 = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, -rhs;
                       bandtol=bandtol0, occupation_threshold, q, kwargs...)  # = -χ04P * rhs
    # TODO Useful printing based on data in res0, i.e.
    # (; δψ, δoccupation, δεF, res.n_iter, res.residual_norms, bandtol, res.converged)
    δρ0 = compute_δρ(basis, ψ, res0.δψ, occupation, res0.δoccupation; occupation_threshold, q)
    res0 = nothing

    # compute total δρ
    function dielectric_adjoint(δρ)
        δV = apply_kernel(basis, δρ; ρ, q)
        # TODO
        # Would be nice to play with abstol / reltol etc. to avoid over-solving
        # for the initial GMRES steps.
        χ0δV = apply_χ0(ham, ψ, occupation, εF, eigenvalues, δV; miniter=1,
                        occupation_threshold, tol_density=tol, bandtolalg, q, kwargs...).δρ
        δρ - χ0δV
    end
    δρ, info_gmres = linsolve(dielectric_adjoint, δρ0;
                              ishermitian=false,
                              tol, verbosity=(verbose ? 3 : 0))
    info_gmres.converged == 0 && @warn "Solve_ΩplusK_split solver not converged"

    # Compute total change in Hamiltonian applied to ψ
    δVind = apply_kernel(basis, δρ; ρ, q)  # Change in potential induced by δρ

    # For phonon calculations, assemble
    #   δHψ_k = δV_{q} · ψ_{k-q}.
    δHψ = multiply_ψ_by_blochwave(basis, ψ, δVind, q) .- rhs

    # Compute total change in eigenvalues
    δeigenvalues = map(ψ, δHψ) do ψk, δHψk
        map(eachcol(ψk), eachcol(δHψk)) do ψnk, δHψnk
            real(dot(ψnk, δHψnk))  # δε_{nk} = <ψnk | δH | ψnk>
        end
    end

    # Compute final orbital response
    # TODO Here we just use what DFTK did before the inexact Krylov business, namely
    #      a fixed Sternheimer tolerance of tol_density / 10. There are probably
    #      smarter things one could do here
    resfinal = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, δHψ;
                           bandtol=factor_final * tol,
                           tol_density=tol, occupation_threshold, q, kwargs...)
    # TODO Useful printing based on data in resfinal, i.e.
    # (; δψ, δoccupation, δεF, res.n_iter, res.residual_norms, bandtol, res.converged)

    (; resfinal.δψ, δρ, δHψ, δVind, δeigenvalues, resfinal.δoccupation, resfinal.δεF, info_gmres)
end

function solve_ΩplusK_split(scfres::NamedTuple, rhs; kwargs...)
    solve_ΩplusK_split(scfres.ham, scfres.ρ, scfres.ψ, scfres.occupation,
                       scfres.εF, scfres.eigenvalues, rhs;
                       scfres.occupation_threshold, scfres.mixing,
                       bandtolalg=BandtolBalanced(scfres), kwargs...)
end
