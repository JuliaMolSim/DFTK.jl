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
    n_components = H.basis.model.n_components
    @assert n_components == 1
    δψ = proj_tangent(δψ, ψ)
    Ωδψ = [H.blocks[ik] * δψk - δψk * Λ[ik]
           for (ik, δψk) in enumerate(blochwaves_as_matrices(δψ))]
    proj_tangent!(blochwave_as_tensor.(Ωδψ, n_components), ψ)
end

"""
    apply_K(basis::PlaneWaveBasis, δψ, ψ, ρ, occupation)

Compute the application of K defined at ψ to δψ. ρ is the density issued from ψ.
δψ also generates a δρ, computed with `compute_δρ`.
"""
# T@D@ basis redundant; change signature maybe?
@views @timing function apply_K(basis::PlaneWaveBasis, δψ, ψ, ρ, occupation)
    δψ = proj_tangent(δψ, ψ)
    δρ = compute_δρ(ψ, δψ, occupation)
    δV = apply_kernel(basis, δρ; ρ)

    Kδψ = map(enumerate(ψ)) do (ik, ψk)
        kpt = basis.kpoints[ik]
        δVψk = similar(ψk)

        for n = 1:size(ψk, 3)
            ψnk_real = ifft(basis, kpt, ψk[:, :, n])
            δVψnk_real = reduce(hcat, δV[:, :, :, kpt.spin] .* ψnk_real[σ, :, :, :]
                                for σ = 1:size(ψk, 1))
            δVψk[:, :, n] = fft(basis, kpt, δVψnk_real)
        end
        δVψk
    end
    # ensure projection onto the tangent space
    proj_tangent!(Kδψ, ψ)
end

"""
    solve_ΩplusK(ψ::BlochWaves{T}, rhs, occupation;
                 tol=1e-10, verbose=false) where {T}

Return δψ where (Ω+K) δψ = rhs
"""
@timing function solve_ΩplusK(ψ::BlochWaves{T}, rhs, occupation; callback=identity,
                              tol=1e-10) where {T}
    basis = ψ.basis
    filled_occ = filled_occupation(basis.model)
    # for now, all orbitals have to be fully occupied -> need to strip them beforehand
    @assert all(all(occ_k .== filled_occ) for occ_k in occupation)

    # To mpi-parallelise we have to deal with the fact that the linear algebra
    # in the CG (dot products, norms) couples k-Points. Maybe take a look at
    # the PencilArrays.jl package to get this done automatically.
    @assert mpi_nprocs() == 1  # Distributed implementation not yet available

    # compute quantites at the point which define the tangent space
    ρ = compute_density(ψ, occupation)
    H = energy_hamiltonian(ψ, occupation; ρ).ham

    ψ_matrices = blochwaves_as_matrices(ψ)
    pack(ψ) = reinterpret_real(pack_ψ(ψ))
    unpack(x) = unpack_ψ(reinterpret_complex(x), size.(ψ))
    unsafe_unpack(x) = unsafe_unpack_ψ(reinterpret_complex(x), size.(ψ))

    # project rhs on the tangent space before starting
    proj_tangent!(rhs, ψ)
    rhs_pack = pack(rhs)

    # preconditioner
    Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    for ik = 1:length(Pks)
        precondprep!(Pks[ik], ψ_matrices[ik])
    end
    function f_ldiv!(x, y)
        δψ = unpack(y)
        proj_tangent!(δψ, ψ)
        Pδψ = [ Pks[ik] \ δψk for (ik, δψk) in enumerate(δψ)]
        # T@D@ revert deepcopy
        Pδψ = deepcopy(proj_tangent!(Pδψ, ψ))
        x .= pack(Pδψ)
    end

    # Rayleigh-coefficients
    Λ = [ψk'Hψk for (ψk, Hψk) in zip(ψ_matrices, H * ψ_matrices)]

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
        # T@D@ revert deepcopy
        δψ = deepcopy(proj_tangent!(δψ, ψ))
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
                            verbose=false, occupation_threshold, kwargs...) where {T}
    # Using χ04P = -Ω^-1, E extension operator (2P->4P) and R restriction operator:
    # (Ω+K)^-1 =  Ω^-1 ( 1 -   K   (1 + Ω^-1 K  )^-1    Ω^-1  )
    #          = -χ04P ( 1 -   K   (1 - χ04P K  )^-1   (-χ04P))
    #          =  χ04P (-1 + E K2P (1 - χ02P K2P)^-1 R (-χ04P))
    # where χ02P = R χ04P E and K2P = R K E
    basis = ham.basis
    @assert size(rhs[1]) == size(ψ[1])  # Assume the same number of bands in ψ and rhs

    ψ_array = denest(ψ)
    # compute δρ0 (ignoring interactions)
    δψ0, δoccupation0 = apply_χ0_4P(ham, ψ_array, occupation, εF, eigenvalues, -rhs;
                                    tol=tol_sternheimer, occupation_threshold,
                                    kwargs...)  # = -χ04P * rhs
    δρ0 = compute_δρ(ψ, δψ0, occupation, δoccupation0; occupation_threshold)

    # compute total δρ
    pack(δρ)   = vec(δρ)
    unpack(δρ) = reshape(δρ, size(ρ))
    function eps_fun(δρ)
        δρ = unpack(δρ)
        δV = apply_kernel(basis, δρ; ρ)
        # TODO
        # Would be nice to play with abstol / reltol etc. to avoid over-solving
        # for the initial GMRES steps.
        χ0δV = apply_χ0(ham, ψ, occupation, εF, eigenvalues, δV;
                        occupation_threshold, tol=tol_sternheimer, kwargs...)
        pack(δρ - χ0δV)
    end
    J = LinearMap{T}(eps_fun, prod(size(δρ0)))
    δρ, history = gmres(J, pack(δρ0); reltol=0, abstol=tol, verbose, log=true)
    δρ = unpack(δρ)

    # Compute total change in Hamiltonian applied to ψ
    δVind = apply_kernel(basis, δρ; ρ)  # Change in potential induced by δρ
    δHψ = @views map(basis.kpoints, ψ, rhs) do kpt, ψk, rhsk
        δVindψk = RealSpaceMultiplication(basis, kpt, δVind[:, :, :, kpt.spin]) * ψk
        δVindψk - rhsk
    end

    # Compute total change in eigenvalues
    δeigenvalues = map(ψ_array, δHψ) do ψk, δHψk
        map(eachslice(ψk; dims=3), eachslice(δHψk; dims=3)) do ψnk, δHψnk
            real(dot(ψnk, δHψnk))  # δε_{nk} = <ψnk | δH | ψnk>
        end
    end

    δψ, δoccupation, δεF = apply_χ0_4P(ham, ψ_array, occupation, εF, eigenvalues, δHψ;
                                       occupation_threshold, tol=tol_sternheimer,
                                       kwargs...)

    (; δψ, δρ, δHψ, δVind, δeigenvalues, δoccupation, δεF, history)
end

function solve_ΩplusK_split(scfres::NamedTuple, rhs; kwargs...)
    solve_ΩplusK_split(scfres.ham, scfres.ρ, scfres.ψ, scfres.occupation,
                       scfres.εF, scfres.eigenvalues, rhs;
                       scfres.occupation_threshold, kwargs...)
end
