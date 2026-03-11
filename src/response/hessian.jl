using KrylovKit
using LinearMaps

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
    map(enumerate(δψ)) do (ik, δψk)
        Ωδψ = H.blocks[ik] * δψk
        mul!(Ωδψ, δψk, Λ[ik], -1, 1)
        proj_tangent_kpt!(Ωδψ, ψ[ik])
        Ωδψ
    end
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
    # normalize here so we can use unnormalized FFTs for extra speed
    δV .*= basis.fft_grid.ifft_normalization * basis.fft_grid.fft_normalization

    ψnk_real = similar(G_vectors(basis), promote_type(T, eltype(ψ[1])))
    Kδψ = map(enumerate(ψ)) do (ik, ψk)
        kpt = basis.kpoints[ik]
        δVψk = similar(ψk)

        for n = 1:size(ψk, 2)
            ifft!(ψnk_real, basis, kpt, ψk[:, n]; normalize=false)
            ψnk_real .*= δV[:, :, :, kpt.spin]
            fft!(δVψk[:, n], basis, kpt, ψnk_real; normalize=false)
        end
        δVψk
    end
    # ensure projection onto the tangent space
    proj_tangent!(Kδψ, ψ)
end

"""
Prepares a function that will perform an optimized application of Ω+K to a vector.
This version trades additional memory for speed by caching `ifft(ψ)` and `ifft(δψ)`.
"""
@views @timing function prepare_ΩplusK(basis::PlaneWaveBasis, H::Hamiltonian,
                                       ψ, ρ, occupation)
    if !(H.blocks[1] isa DftHamiltonianBlock) || !isnothing(H.blocks[1].divAgrad_op)
        error("prepare_ΩplusK only implemented for DftHamiltonianBlock without divAgrad for now")
    end

    Λ = [ψk'Hψk for (ψk, Hψk) in zip(ψ, H * ψ)]
    # use unnormalized plans for extra speed, normalize at the end
    fft_normalization = basis.fft_grid.fft_normalization
    ifft_normalization = basis.fft_grid.ifft_normalization
    ψ_real = map(basis.kpoints, ψ) do kpt, ψk
        [ifft(basis, kpt, ψk[:, n]; normalize=false) for n in 1:size(ψk, 2)]
    end
    δψ_real = deepcopy(ψ_real)

    @timing function apply_ΩplusK(δψ)
        δψ = proj_tangent(δψ, ψ)
        @timing "ifft(δψ)" begin
            for (kpt, δψk, δψk_real) in zip(basis.kpoints, δψ, δψ_real)
                for n = 1:size(δψk, 2)
                    ifft!(δψk_real[n], basis, kpt, δψk[:, n]; normalize=false)
                end
            end
        end
        # Computation of δρ with cached ifft(ψ) and ifft(δψ)
        @timing "compute δρ" begin
            δρ = zeros_like(ρ)
            for (ik, kpt) in enumerate(basis.kpoints), n in 1:size(ψ[ik], 2)
                ψnk_real = ψ_real[ik][n]
                δψnk_real = δψ_real[ik][n]
                # use unnormalized plans for extra speed, normalize at the end
                δρ[:, :, :, kpt.spin] .+= ifft_normalization^2 .* real.(
                    2 .* occupation[ik][n] .* basis.kweights[ik]
                      .* conj.(ψnk_real) .* δψnk_real)
            end
            mpi_sum!(δρ, basis.comm_kpts)
            δρ = symmetrize_ρ(basis, δρ; do_lowpass=false)
        end
        # Compute kernel
        δV = apply_kernel(basis, δρ; ρ)
        # normalize here so we can use unnormalized FFTs for extra speed
        δV .*= fft_normalization * ifft_normalization
        # Now apply both δV * ψ (K) and the first part of H * δψ (first part of Ω),
        # to share the fft of the summed result
        result = [zero(ψk) for ψk in ψ]
        res_real = zeros_like(ψ[1], basis.fft_size...)
        for (ik, kpt) in enumerate(basis.kpoints)
            Hk = H.blocks[ik]::DftHamiltonianBlock
            potential = Hk.local_op.potential .* fft_normalization .* ifft_normalization

            for iband in 1:size(ψ[ik], 2)
                @timing "δV * ψ + local * δψ" begin
                    res_real .= (  potential .* δψ_real[ik][iband]
                                .+ δV[:, :, :, kpt.spin] .* ψ_real[ik][iband])
                    fft!(result[ik][:, iband], basis, kpt, res_real; normalize=false)
                end
            end

            result[ik] .+= Hk.fourier_op.multiplier .* δψ[ik]
            if !isnothing(Hk.nonlocal_op)
                @timing "nonlocal" begin
                    apply!((; fourier=result[ik], real=nothing),
                            Hk.nonlocal_op,
                            (; fourier=δψ[ik], real=nothing))
                end
            end
        end
        # Finish with -ψ * Λ part of Ω and projection onto the tangent space
        map(enumerate(δψ)) do (ik, δψk)
            mul!(result[ik], δψk, Λ[ik], -1, 1)
            proj_tangent_kpt!(result[ik], ψ[ik])
            result[ik]
        end
    end
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
    mpi_master(info.basis.comm_kpts) || return info  # Only print on master

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

    resnorm = @sprintf "%20.2f" log10(only(info.residual_norms))
    time = @sprintf "% 6s" TimerOutputs.prettytime(runtime_ns)
    @printf "% 3d   %s   %s\n" info.n_iter resnorm time
    flush(stdout)
    info
end

"""
Solve density-functional perturbation theory problem,
that is return δψ where (Ω+K) δψ = -δHextψ.
"""
@timing function solve_ΩplusK(basis::PlaneWaveBasis{T}, ψ, δHextψ, occupation;
                              callback=ResponseCallback(), tol=1e-10) where {T}
    # for now, all orbitals have to be fully occupied -> need to strip them beforehand
    check_full_occupation(basis, occupation)

    # compute quantites at the point which define the tangent space
    ρ = compute_density(basis, ψ, occupation)
    H = energy_hamiltonian(basis, ψ, occupation; ρ).ham

    # K is not C-linear, so we work in R^2N instead of C^N via the pack/unpack routines
    pack(ψ) = reinterpret_real(pack_ψ(ψ))
    unpack(x) = unpack_ψ(reinterpret_complex(x), size.(ψ))
    unsafe_unpack(x) = unsafe_unpack_ψ(reinterpret_complex(x), size.(ψ))

    # project δHextψ on the tangent space before starting
    proj_tangent!(δHextψ, ψ)
    δHextψ_pack = pack(δHextψ)

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

    # mapping of the linear system on the tangent space
    apply_ΩpK = prepare_ΩplusK(basis, H, ψ, ρ, occupation)

    function ΩpK(x)
        δψ = unsafe_unpack(x)
        res = apply_ΩpK(δψ)
        pack(res)
    end
    J = LinearMap{T}(ΩpK, size(δHextψ_pack, 1))

    # solve (Ω+K) δψ = -δHextψ on the tangent space with CG
    function proj!(Px, x)
        δψ = unpack(x)
        proj_tangent!(δψ, ψ)
        Px .= pack(δψ)
    end
    # custom inner product that Ω+K is self-adjoint with respect to
    function weighted_dots(x, y)
        δψx = unsafe_unpack(x)
        δψy = unsafe_unpack(y)
        # real(dot) here because we work in R^2N rather than C^N
        [weighted_ksum(basis, [real(dot(δψx[ik], δψy[ik])) for ik in 1:length(basis.kpoints)])]
    end
    res = cg(J, -δHextψ_pack; precon=FunctionPreconditioner(f_ldiv!), proj!,
             tol=tol, callback=info -> callback(merge(info, (; basis=basis))),
             my_columnwise_dots=weighted_dots)
    (; δψ=unpack(res.x), res.converged, res.tol, res.residual_norms,
     res.n_iter)
end

struct OmegaPlusKDefaultCallback
    show_time::Bool
    show_σmin::Bool
    prev_time::Ref{UInt64}
end
function OmegaPlusKDefaultCallback(; show_σmin=false, show_time=true)
    OmegaPlusKDefaultCallback(show_time, show_σmin, Ref(zero(UInt64)))
end
function (cb::OmegaPlusKDefaultCallback)(info)
    io = stdout
    # Default to MPI.COMM_WORLD for logging if basis not provided
    comm = MPI.COMM_WORLD
    haskey(info, :basis) && (comm = info.basis.comm_kpts)

    avgCG = 0.0
    if haskey(info, :Axinfos) && haskey(first(info.Axinfos), :n_iter)
        # Axinfo: NamedTuple returned by mul_inexact(::DielectricAdjoint, ...)
        # Axinfos is the collection of all these named tuples since the last callback

        # Sum all CG iterations over all bands and all Axinfos, average over k-points
        avgCG = sum(info.Axinfos) do Axinfo
            mean(sum, Axinfo.n_iter)
        end
        avgCG = mpi_mean(avgCG, comm)
    end

    !mpi_master(comm) && return info  # Rest is printing => only do on master

    show_time  = (hasproperty(info, :runtime_ns) && cb.show_time)
    label_time = show_time    ? ("  Δtime ", "  ------", " "^8) : ("", "", "")
    label_s    = cb.show_σmin ? ("  ≈σ_min", "  ------", " "^8) : ("", "", "")

    tstr = ""
    if show_time
        # Clear a potentially previously cached time
        info.stage == :noninteracting && (cb.prev_time[] = 0)
        tstr = @sprintf "  % 6s" TimerOutputs.prettytime(info.runtime_ns - cb.prev_time[])
        cb.prev_time[] = info.runtime_ns
    end

    if info.stage == :noninteracting
        # Non-interacting run before the main Dyson equation solve
        println(io, "Iter  Restart  Krydim", label_s[1], "  log10(res)  avg(CG)",
                label_time[1], "  Comment")
        @printf(io, "%s  %s  %s%s  %s  %s%s  %s\n",
                "-"^4, "-"^7, "-"^6, label_s[2], "-"^10, "-"^7, label_time[2], "-"^15)
        @printf(io, "%21s%s  %10s  %7.1f%s  %15s\n",
                "", label_s[3], "", avgCG, tstr, "Non-interacting")
    elseif info.stage == :iterate
        n_iter  = info.n_iter
        resstr  = format_log8(info.resid_history[n_iter])
        comment = ((n_iter-1) in info.restart_history) ? "Restart" : ""
        sstr    = cb.show_σmin ? (@sprintf "  %6.2f" info.s) : ""
        restart = length(info.restart_history)
        @printf(io, "%4i  %7i  %6i%s  %10s  %7.1f%s  %s\n",
                n_iter, restart, info.k, sstr, resstr, avgCG, tstr, comment)
    elseif info.stage == :final
        @printf(io, "%21s%s  %10s  %7.1f%s  %s\n",
                "", label_s[3], "", avgCG, tstr, "Final orbitals")
    end
    flush(stdout)
    info
end

"""
Solve the problem `(Ω+K) δψ = -δHextψ` (density-functional perturbation theory)
using a split algorithm, where
`δψ` is the total variation in the orbitals `ψ` corresponding to the external perturbation δHext.
Additionally returns:
    - `δρ`:  Total variation in density
    - `δHψ`: Total variation in Hamiltonian applied to orbitals
    - `δeigenvalues`: Total variation in eigenvalues
    - `δVind`: Change in potential induced by `δρ` (the term needed on top of `δHextψ`
      to get `δHψ`).

Input parameters:
- `tol`: Desired tolerance in the density variation and orbital variation
- `mixing`: Mixing to use to precondition GMRES
- `maxiter_sternheimer`: Maximal number of iterations for each of the Sternheimer solvers
- `maxiter`: Maximal number of iterations for the Dyson equation solver
   convergence, it will automatically stop.
- `krylovdim`: Maximal Krylov subspace dimension in the Dyson equation solver
- `s`: Initial guess for the smallest singular value of the upper Hessenberg matrix
   in the Dyson equation solver.
- `bandtolalg`: Algorithm for adaptive selection of Sternheimer tolerances
   in the Dyson equation solver,
   see [arxiv 2505.02319](https://arxiv.org/pdf/2505.02319) for more details.
"""
@timing function solve_ΩplusK_split(ham::Hamiltonian, ρ::AbstractArray{T}, ψ, occupation, εF,
                                    eigenvalues, δHextψ;
                                    δtemperature=zero(real(T)),
                                    tol=1e-8, verbose=true,
                                    mixing=SimpleMixing(),
                                    occupation_threshold,
                                    bandtolalg=BandtolBalanced(ham.basis, ψ, occupation; occupation_threshold),
                                    factor_initial=1/10, factor_final=1/10,
                                    q=zero(Vec3{real(T)}),
                                    maxiter_sternheimer=100,
                                    maxiter=100, krylovdim=20, s=100,
                                    callback=verbose ? OmegaPlusKDefaultCallback() : identity,
                                    kwargs...) where {T}
    # TODO mixing=LdosMixing(; adjust_temperature=UseScfTemperature()) would be a better
    #      default in theory, but does not work out of the box, so not done for now
    # TODO Debug why and enable LdosMixing by default
    if !(mixing isa SimpleMixing || mixing isa KerkerMixing || mixing isa KerkerDosMixing)
        @warn("solve_ΩplusK_split has only been tested with one of SimpleMixing, " *
              "KerkerMixing or KerkerDosMixing")
    end

    # Using χ04P = -Ω^-1, E extension operator (2P->4P) and R restriction operator:
    # (Ω+K)^-1 =  Ω^-1 ( 1 -   K   (1 + Ω^-1 K  )^-1    Ω^-1  )
    #          = -χ04P ( 1 -   K   (1 - χ04P K  )^-1   (-χ04P))
    #          =  χ04P (-1 + E K2P (1 - χ02P K2P)^-1 R (-χ04P))
    # where χ02P = R χ04P E and K2P = R K E
    basis = ham.basis
    @assert size(δHextψ[1]) == size(ψ[1])
    start_ns = time_ns()

    # TODO Use tol_density=tol/10 to make sure that the density is very accurate.
    #      This is likely overdoing it and we should investigate if a smaller
    #      value also does the trick.

    # compute δρ0 (ignoring interactions)
    δρ0, δψ0 = let  # Make sure memory owned by res0 is freed
        res0 = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, δHextψ;
                           δtemperature,
                           maxiter=maxiter_sternheimer, tol=tol * factor_initial,
                           bandtolalg, occupation_threshold,
                           q, kwargs...)  # = χ04P * δHext
        callback((; stage=:noninteracting, runtime_ns=time_ns() - start_ns, basis,
                    Axinfos=[(; tol=tol*factor_initial, res0...)]))
        (compute_δρ(basis, ψ, res0.δψ, occupation, res0.δoccupation;
                    occupation_threshold, q), res0.δψ)
    end

    # compute total δρ
    # TODO Can be smarter here, e.g. use mixing to come up with initial guess.
    ε_adj = DielectricAdjoint(ham, ρ, ψ, occupation, εF, eigenvalues, occupation_threshold,
                              bandtolalg, maxiter_sternheimer, q)
    precon = FunctionPreconditioner() do Pδρ, δρ
        Pδρ .= vec(mix_density(mixing, basis, reshape(δρ, size(ρ));
                               ham, basis, ρin=ρ, εF, eigenvalues, ψ))
    end
    callback_inner(info) = callback(merge(info, (; runtime_ns=time_ns() - start_ns, basis=basis)))
    info_gmres = inexact_gmres(ε_adj, vec(δρ0);
                               tol, precon, krylovdim, maxiter, s,
                               callback=callback_inner, kwargs...)
    δρ = reshape(info_gmres.x, size(ρ))
    if !info_gmres.converged
        @warn "Solve_ΩplusK_split solver not converged"
    end

    # Now we got δρ, but we're not done yet, because we want the full output of the four-point apply_χ0_4P,
    # so we redo an apply_χ0_4P

    # Induced potential variation
    δVind = apply_kernel(basis, δρ; ρ, q)  # Change in potential induced by δρ

    # Total variation δHtot ψ
    # For phonon calculations, assemble
    #   δHψ_k = δV_{q} · ψ_{k-q}.
    δHtotψ = multiply_ψ_by_blochwave(basis, ψ, δVind, q) .+ δHextψ

    # Compute final orbital response
    # TODO Here we just use what DFTK did before the inexact Krylov business, namely
    #      a fixed Sternheimer tolerance of tol / 10. There are probably
    #      smarter things one could do here
    resfinal = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, δHtotψ;
                           δtemperature,
                           maxiter=maxiter_sternheimer, tol=tol * factor_final,
                           bandtolalg, occupation_threshold, q, δψ0, kwargs...)
    callback((; stage=:final, runtime_ns=time_ns() - start_ns, basis,
                Axinfos=[(; tol=tol*factor_final, resfinal...)]))
    # Compute total change in eigenvalues
    δeigenvalues = map(ψ, δHtotψ) do ψk, δHtotψk
        map(eachcol(ψk), eachcol(δHtotψk)) do ψnk, δHtotψnk
            real(dot(ψnk, δHtotψnk))  # δε_{nk} = <ψnk | δHtot | ψnk>
        end
    end

    (; resfinal.δψ, δρ, δHtotψ, δVind, δρ0, δeigenvalues, resfinal.δoccupation,
       resfinal.δεF, ε_adj, info_gmres)
end

function solve_ΩplusK_split(scfres::NamedTuple, δHextψ; kwargs...)
    if (scfres.mixing isa KerkerMixing || scfres.mixing isa KerkerDosMixing)
        mixing = scfres.mixing
    else
        mixing = SimpleMixing()
    end
    solve_ΩplusK_split(scfres.ham, scfres.ρ, scfres.ψ, scfres.occupation,
                       scfres.εF, scfres.eigenvalues, δHextψ;
                       scfres.occupation_threshold, mixing,
                       bandtolalg=BandtolBalanced(scfres), kwargs...)
end

struct DielectricAdjoint{Tρ, Tψ, Toccupation, TεF, Teigenvalues, Tq}
    ham::Hamiltonian
    ρ::Tρ
    ψ::Tψ
    occupation::Toccupation
    εF::TεF
    eigenvalues::Teigenvalues
    occupation_threshold::Float64
    bandtolalg
    maxiter::Int  # CG maximum number of iterations
    q::Tq
end

@doc raw"""
Representation of the dielectric adjoint operator ``ε^† = (1 - χ_0 K)``.
This is the adjoint of the dielectric operator ``ε = (1 - K χ_0)``.
"""
function DielectricAdjoint(scfres; bandtolalg=BandtolBalanced(scfres), q=zero(Vec3{Float64}), maxiter=100)
    DielectricAdjoint(scfres.ham, scfres.ρ, scfres.ψ, scfres.occupation, scfres.εF,
                      scfres.eigenvalues, scfres.occupation_threshold, bandtolalg, maxiter, q)
end
@timing "DielectricAdjoint" function mul_approximate(ε_adj::DielectricAdjoint, δρ; rtol=0.0, kwargs...)
    δρ = reshape(δρ, size(ε_adj.ρ))
    basis = ε_adj.ham.basis
    δV = apply_kernel(basis, δρ; ε_adj.ρ, ε_adj.q)
    res = apply_χ0(ε_adj.ham, ε_adj.ψ, ε_adj.occupation, ε_adj.εF, ε_adj.eigenvalues, δV;
                   miniter=1, ε_adj.occupation_threshold, tol=rtol*norm(δρ),
                   ε_adj.bandtolalg, ε_adj.q, ε_adj.maxiter, kwargs...)
    χ0δV = res.δρ
    Ax = vec(δρ - χ0δV)  # (1 - χ0 K) δρ
    (; Ax, info=(; rtol, res...))
end
function size(ε_adj::DielectricAdjoint, i::Integer)
    if 1 ≤ i ≤ 2
        return prod(size(ε_adj.ρ))
    else
        return one(i)
    end
end
