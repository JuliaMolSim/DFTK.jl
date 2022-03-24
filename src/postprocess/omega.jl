"""
    apply_Ω(δψ, ψ, H::Hamiltonian, Λ)

Compute the application of Ω defined at ψ to δψ. H is the Hamiltonian computed
from ψ and Λ is the set of Rayleigh coefficients ψk' * Hk * ψk at each k-point.
"""
function apply_Ω(δψ, ψ, H::Hamiltonian, Λ)
    δψ = proj_tangent(δψ, ψ)
    Ωδψ = [H.blocks[ik] * δψk - δψk * Λ[ik] for (ik, δψk) in enumerate(δψ)]
    proj_tangent!(Ωδψ, ψ)
end

"""
    apply_K(basis::PlaneWaveBasis, δψ, ψ, ρ, occupation)

Compute the application of K defined at ψ to δψ. ρ is the density issued from ψ.
δψ also generates a δρ, computed with `compute_δρ`.
"""
@views function apply_K(basis::PlaneWaveBasis, δψ, ψ, ρ, occupation)
    δψ = proj_tangent(δψ, ψ)
    δρ = compute_δρ(basis, ψ, δψ, occupation)
    δV = apply_kernel(basis, δρ; ρ=ρ)

    Kδψ = map(enumerate(ψ)) do (ik, ψk)
        kpt = basis.kpoints[ik]
        δVψk = similar(ψk)

        for n = 1:size(ψk, 2)
            ψnk_real = G_to_r(basis, kpt, ψk[:, n])
            δVψnk_real = δV[:, :, :, kpt.spin] .* ψnk_real
            δVψk[:, n] = r_to_G(basis, kpt, δVψnk_real)
        end
        δVψk
    end
    # ensure projection onto the tangent space
    proj_tangent!(Kδψ, ψ)
end

"""
    solve_ΩplusK(basis::PlaneWaveBasis{T}, ψ, res, occupation;
                 tol_cg=1e-10, verbose=false) where T

Return δψ where (Ω+K) δψ = rhs
"""
function solve_ΩplusK(basis::PlaneWaveBasis{T}, ψ, rhs, occupation;
                      tol_cg=1e-10, verbose=false) where T
    @assert mpi_nprocs() == 1  # Distributed implementation not yet available
    filled_occ = filled_occupation(basis.model)
    # for now, all orbitals have to be fully occupied -> need to strip them beforehand
    @assert all(all(occ_k .== filled_occ) for occ_k in occupation)

    # compute quantites at the point which define the tangent space
    ρ = compute_density(basis, ψ, occupation)
    _, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)

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
    δψ, history = cg(J, rhs_pack, Pl=FunctionPreconditioner(f_ldiv!),
                  reltol=0, abstol=tol_cg, verbose=verbose, log=true)

    (; δψ=unpack(δψ), history)
end


# Solves Ω+K using a split algorithm
# With χ04P = -Ω^-1,
# (Ω+K)^-1 = Ω^-1 (1 - K(1+Ω^-1 K)^-1 Ω^-1)
# (Ω+K)^-1 = -χ04P (1 + K(1 - χ04P K)^-1 χ04P)
# (Ω+K)^-1 = -χ04P (1 + E K2P (1 - χ02P K2P)^-1 R χ04P)
# where χ02P = R χ04P E and K2P = R K E
function solve_ΩplusK_split(ham::Hamiltonian, ρ::AbstractArray{T}, ψ, occupation, εF,
                            eigenvalues, rhs;
                            tol_dyson=1e-8, tol_cg=1e-12, verbose=false, kwargs...) where T
    basis = ham.basis

    # compute δρ0 (ignoring interactions)
    δψ0 = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, rhs; tol_cg, verbose, kwargs...)
    δρ0 = compute_δρ(basis, ψ, δψ0, occupation)

    pack(δρ)   = vec(δρ)
    unpack(δρ) = reshape(δρ, size(ρ))
    # compute total δρ
    function eps_fun(δρ)
        δρ = unpack(δρ)
        δV = apply_kernel(basis, δρ; ρ)
        χ0δV = apply_χ0(ham, ψ, εF, eigenvalues, δV; tol_cg, kwargs...)
        pack(δρ - χ0δV)
    end
    J = LinearMap{T}(eps_fun, length(pack(δρ0)))
    δρ, history = gmres(J, pack(δρ0); reltol=0, abstol=tol_dyson, verbose, log=true)
    δV = apply_kernel(basis, unpack(δρ); ρ)

    δVψ = [DFTK.RealSpaceMultiplication(basis, kpt, @views δV[:, :, :, kpt.spin]) * ψ[ik]
           for (ik, kpt) in enumerate(basis.kpoints)]
    δψ = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, δVψ; tol_cg, verbose, kwargs...)
    .- (δψ0 .+ δψ), history

end

function solve_ΩplusK_split(basis::PlaneWaveBasis, ψ, rhs, occupation; kwargs...)
    ρ = compute_density(basis, ψ, occupation)
    _, H = energy_hamiltonian(basis, ψ, occupation; ρ)

    eigenvalues = [real.(eigvals(ψk'Hψk)) for (ψk, Hψk) in zip(ψ, H * ψ)]
    occupation, εF = compute_occupation(basis, eigenvalues)

    solve_ΩplusK_split(H, ρ, ψ, occupation, εF, eigenvalues, rhs; kwargs...)
end

function solve_ΩplusK_split(scfres::NamedTuple, rhs; kwargs...)
    n_ep_extra = scfres.n_ep_extra

    ψ       = [@view ψk[:, 1:end-n_ep_extra]     for ψk in scfres.ψ]
    ψ_extra = [@view ψk[:, end-n_ep_extra+1:end] for ψk in scfres.ψ]

    evals   = [εk[1:end-n_ep_extra]     for εk in scfres.eigenvalues]
    ε_extra = [εk[end-n_ep_extra+1:end] for εk in scfres.eigenvalues]

    occupation = [occk[1:end-n_ep_extra]     for occk in scfres.occupation]
    occ_extra  = [occk[end-n_ep_extra+1:end] for occk in scfres.occupation]

    @assert length(rhs) == length(ψ)
    rhs_capped = map(zip(ψ, rhs)) do (ψk, rhsk)
        n_bands = size(ψk, 2)
        @assert size(rhsk, 2) >= n_bands
        @view rhsk[:, 1:n_bands]
    end
    solve_ΩplusK_split(scfres.ham, scfres.ρ, ψ, occupation, scfres.εF, evals, rhs_capped;
                       ψ_extra, ε_extra, kwargs...)
end
