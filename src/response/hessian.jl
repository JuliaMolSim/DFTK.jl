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
                 tol=1e-10, verbose=false) where T

Return δψ where (Ω+K) δψ = rhs
"""
function solve_ΩplusK(basis::PlaneWaveBasis{T}, ψ, rhs, occupation;
                      tol=1e-10, verbose=false) where T
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
                  reltol=0, abstol=tol, verbose=verbose, log=true)

    (; δψ=unpack(δψ), history)
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
function solve_ΩplusK_split(ham::Hamiltonian, ρ::AbstractArray{T}, ψ, occupation, εF,
                            eigenvalues, rhs; tol=1e-8, tol_sternheimer=tol/10,
                            verbose=false, occupation_threshold, mixing=LdosMixing(),
                            kwargs...) where T
    # Using χ04P = -Ω^-1, E extension operator (2P->4P) and R restriction operator:
    # (Ω+K)^-1 =  Ω^-1 ( 1 -   K   (1 + Ω^-1 K  )^-1    Ω^-1  )
    #          = -χ04P ( 1 -   K   (1 - χ04P K  )^-1   (-χ04P))
    #          =  χ04P (-1 + E K2P (1 - χ02P K2P)^-1 R (-χ04P))
    # where χ02P = R χ04P E and K2P = R K E
    basis = ham.basis
    @assert size(rhs[1]) == size(ψ[1])  # Assume the same number of bands in ψ and rhs

    # compute δρ0 (ignoring interactions)
    δψ0 = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, -rhs;
                      reltol=0, abstol=tol_sternheimer,
                      occupation_threshold, kwargs...)  # = -χ04P * rhs
    δρ0 = compute_δρ(basis, ψ, δψ0, occupation)

    # Remove DC component (we add it back on later)
    DC_δρ0 = mean(δρ0)
    δρ0 .-= DC_δρ0

    # Compute δρ0
    pack(δρ)   = vec(δρ)
    unpack(δρ) = reshape(δρ, size(ρ))
    δρ = pack(zero(δρ0))
    initially_zero = true
    DFTK.reset_timer!(DFTK.timer)

    # Config
    dynamic_tolerance = false
    dynamic_anderson = false
    innertol = Ref(1e-6)

    fixed_anderson = true

    initial_solve = false
    initial_tol_sternheimer = 1e-3
    # end


    function dynamic_adjoint()
        LinearMap{T}(prod(size(δρ0))) do δρ0
            δρ0 .-= mean(δρ0)
            δρ0 = unpack(δρ0)
            δV = apply_kernel(basis, δρ0; ρ)
            χ0δV = apply_χ0(ham, ψ, occupation, εF, eigenvalues, δV;
                            occupation_threshold, abstol=innertol[] / norm(δV), reltol=zero(T),
                            kwargs...)
            δρ = symmetrize_ρ(basis, δρ0 - χ0δV)
            δρ .-= mean(δρ)
            pack(δρ)
        end
    end

    Pinv = I

    if dynamic_anderson
        δρ = let
            β = 0.6
            maxiter = 100
            x0 = pack(δρ)
            ε = dynamic_adjoint()
            b = pack(δρ0)
            fp_solver = scf_nlsolve_solver(; m=10, method=:anderson)
            # fp_solver = scf_anderson_solver(; m=10)
            i = 0

            res = fp_solver(x0, maxiter; tol) do x
                i += 1
                residual = Pinv * (b - ε * x)
                rnorm = norm(residual)
                innertol[] = max(tol_sternheimer, min(innertol[], rnorm / 100))
                println(i, "   ", rnorm,  "   ", innertol[])
                x + β * residual
            end
            unpack(res.x)
        end
        initially_zero = false
    end

    if dynamic_tolerance
        iterable = IterativeSolvers.gmres_iterable!(δρ, dynamic_adjoint(), pack(δρ0);
                                                    reltol=0, abstol=tol, restart=10, initially_zero)
        for (i, residual) in enumerate(iterable)
            innertol[] = max(tol_sternheimer, min(innertol[], residual / 100))
            println(i, "   ", residual,  "   ", innertol[])
        end
        δρ = iterable.x
        initially_zero = false
    end

    # compute total δρ by solving ε^† δρ = δρ0 with ε^† = (1 - χ₀ K)
    function dielectric_adjoint(abstol)
        LinearMap{T}(prod(size(δρ0))) do δρ0
            δρ0 .-= mean(δρ0)
            δρ0 = unpack(δρ0)
            δV = apply_kernel(basis, δρ0; ρ)
            χ0δV = apply_χ0(ham, ψ, occupation, εF, eigenvalues, δV;
                            occupation_threshold, abstol, reltol=zero(T),
                            kwargs...)
            δρ = symmetrize_ρ(basis, δρ0 - χ0δV)
            δρ .-= mean(δρ)
            pack(δρ)
        end
    end

    # Crude initial solve using rough Sternheimer tolerance to get good initial guess
    if initial_solve
        δρ = IterativeSolvers.gmres!(δρ, dielectric_adjoint(initial_tol_sternheimer/10),
                                     pack(δρ0); reltol=0, abstol=initial_tol_sternheimer,
                                     initially_zero, verbose)
        initially_zero = false
    end

    if fixed_anderson
        δρ = let
            β = 0.6
            maxiter = 100
            x0 = pack(δρ)
            ε = dielectric_adjoint(tol_sternheimer)
            b = pack(δρ0)
            fp_solver = scf_nlsolve_solver(; m=10, method=:anderson)
            # fp_solver = scf_anderson_solver(; m=10)
            i = 0

            res = fp_solver(x0, maxiter; tol) do x
                i += 1
                residual = Pinv * (b - ε * x)
                rnorm = norm(residual)
                rnorm = norm(residual)
                println(i, "   ", rnorm)
                x + β * residual
            end
            unpack(res.x)
        end
        initially_zero = false
    end


    # Full solve to desired target tolerance
    δρ, history = IterativeSolvers.gmres!(δρ, dielectric_adjoint(tol_sternheimer), pack(δρ0);
                                          reltol=0, abstol=tol, verbose, initially_zero, log=true)
    println(DFTK.timer)

    δρ = unpack(δρ)
    δρ .+= DC_δρ0  # Set DC from δρ0

    # Compute total change in Hamiltonian applied to ψ
    δVind = apply_kernel(basis, δρ; ρ)  # Change in potential induced by δρ
    δHψ = @views map(basis.kpoints, ψ, rhs) do kpt, ψk, rhsk
        δVindψk = RealSpaceMultiplication(basis, kpt, δVind[:, :, :, kpt.spin]) * ψk
        δVindψk - rhsk
    end

    # Compute total change in eigenvalues
    δeigenvalues = map(ψ, δHψ) do ψk, δHψk
        map(eachcol(ψk), eachcol(δHψk)) do ψnk, δHψnk
            real(dot(ψnk, δHψnk))  # δε_{nk} = <ψnk | δH | ψnk>
        end
    end

    δψ = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, δHψ;
                     occupation_threshold, abstol=tol_sternheimer, reltol=0, kwargs...)
    (; δψ, δρ, δHψ, δVind, δeigenvalues, history)
end

function solve_ΩplusK_split(basis::PlaneWaveBasis, ψ, rhs, occupation; kwargs...)
    ρ = compute_density(basis, ψ, occupation)
    _, H = energy_hamiltonian(basis, ψ, occupation; ρ)

    eigenvalues = [real.(eigvals(ψk'Hψk)) for (ψk, Hψk) in zip(ψ, H * ψ)]
    occupation_threshold = kwargs.occupation_threshold
    occupation, εF = compute_occupation(basis, eigenvalues; occupation_threshold)

    solve_ΩplusK_split(H, ρ, ψ, occupation, εF, eigenvalues, rhs; kwargs...)
end

function solve_ΩplusK_split(scfres::NamedTuple, rhs; kwargs...)
    @assert scfres.algorithm == "SCF"  # Otherwise no mixing field.
    solve_ΩplusK_split(scfres.ham, scfres.ρ, scfres.ψ, scfres.occupation,
                       scfres.εF, scfres.eigenvalues, rhs;
                       scfres.occupation_threshold, scfres.mixing, kwargs...)
end
