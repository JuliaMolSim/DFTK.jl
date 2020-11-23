# This file contains functions to perform newton step and a newton solver for
# DFTK problems.


## pack and unpack routines
function packing_routines(basis::PlaneWaveBasis{T}, ψ) where T

    # necessary quantities
    Nk = length(basis.kpoints)

    lengths = [length(ψ[ik]) for ik = 1:Nk]
    starts = copy(lengths)
    starts[1] = 1
    for ik = 1:Nk-1
        starts[ik+1] = starts[ik] + lengths[ik]
    end
    pack(φ) = vcat(Base.vec.(φ)...)
    unpack(x) = [@views reshape(x[starts[ik]:starts[ik]+lengths[ik]-1], size(ψ[ik]))
                 for ik = 1:Nk]

    packed_proj!(ϕ,φ) = proj!(unpack(ϕ), unpack(φ))
    (pack, unpack, packed_proj!)
end

# we compute the residual associated to a set of planewave φ, that is to say
# H(φ)*φ - λ.*φ where λ is the set of rayleigh coefficients associated to the
# φ
# we also return the egval set for further computations
function compute_residual(basis::PlaneWaveBasis{T}, φ, occ) where T

    # necessary quantities
    Nk = length(basis.kpoints)
    ρ = compute_density(basis, φ, occ)
    energies, H = energy_hamiltonian(basis, φ, occ; ρ=ρ[1])
    egval = [ zeros(Complex{T}, size(occ[i])) for i = 1:length(occ) ]

    # compute residual
    res = similar(φ)
    for ik = 1:Nk
        φk = φ[ik]
        N = size(φk, 2)
        Hk = H.blocks[ik]
        # eigenvalues as rayleigh coefficients
        egvalk = [φk[:,i]'*(Hk*φk[:,i]) for i = 1:N]
        # compute residual at given kpoint as H(φ)φ - λφ
        rk = Hk*φk - hcat([egvalk[i] * φk[:,i] for i = 1:N]...)
        egval[ik] = egvalk
        res[ik] = rk
    end

    # return residual after projection onto the tangent space
    (res=proj!(res, φ), ρ, H, egval)
end


# perform a newton step : we take as given a planewave set φ and we return the
# newton step φ - δφ (after proper orthonormalization) where δφ solves Jac * δφ = res
function newton_step(basis::PlaneWaveBasis{T}, φ, res, ρ, H, egval, occ,
                     packing) where T

    # necessary quantities
    Nk = length(basis.kpoints)
    pack, unpack, packed_proj! = packing
    ortho(ψk) = Matrix(qr(ψk).Q)

    # solve linear system with KrlyovKit
    function f(x)
        δφ = unpack(x)
        δφ = proj!(δφ, φ)
        ΩpKx = ΩplusK(basis, δφ, φ, ρ[1], H, egval, occ)
        ΩpKx = proj!(ΩpKx, φ)
        pack(ΩpKx)
    end
    δφ, info = linsolve(f, pack(res);
                        tol=1e-15, verbosity=1,
                        orth=OrthogonalizeAndProject(packed_proj!, pack(φ)))
    δφ = unpack(δφ)
    δφ = proj!(δφ, φ)

    for ik = 1:Nk
        φk = φ[ik]
        δφk = δφ[ik]
        N = size(φk,2)
        for i = 1:N
            φk[:,i] = φk[:,i] - δφk[:,i]
        end
        φk = ortho(φk)
        φ[ik] = φk
    end
    φ
end

# newton algorithm
function newton(basis::PlaneWaveBasis{T}; ψ0=nothing,
                tol=1e-6, max_iter=100) where T

    ## setting parameters
    model = basis.model
    @assert model.spin_polarization in (:none, :spinless)
    @assert model.temperature == 0 # temperature is not yet supported
    filled_occ = DFTK.filled_occupation(model)
    N = div(model.n_electrons, filled_occ)

    ## number of kpoints
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

    ## starting point and orthonormalization routine
    if ψ0 === nothing
        ortho(ψk) = Matrix(qr(ψk).Q)
        ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
              for kpt in basis.kpoints]
    end

    ## packing routines to pack vectors for KrylovKit solver
    packing = packing_routines(basis, ψ0)

    ## error list for convergence plots
    err_list = []
    err_ref_list = []
    k_list = []

    err = 1
    k = 0

    # orbitals to be updated along the iterations
    φ = deepcopy(ψ0)

    while err > tol && k < max_iter
        k += 1
        println("Iteration $(k)...")
        append!(k_list, k)

        # compute next step
        res, ρ, H, egval = compute_residual(basis, φ, occupation)
        φ = newton_step(basis, φ, res, ρ, H, egval, occupation, packing)

        # compute error on the norm
        ρ_next = compute_density(basis, φ, occupation)
        err = norm(ρ_next[1].real - ρ[1].real)
        append!(err_list, err)
        append!(err_ref_list, norm(ρ_next[1].real - scfres.ρ.real))
    end

    # plot results
    figure()
    semilogy(k_list, err_list, "x-", label="|ρ^{k+1} - ρ^k|")
    semilogy(k_list, err_ref_list, "x-", label="|ρ^k - ρref|")
    xlabel("iterations")
    legend()
end


