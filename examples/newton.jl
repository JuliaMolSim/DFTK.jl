# Very basic setup, useful for testing
using DFTK
using LinearAlgebra
using PyPlot

# import aux file
include("aposteriori_operators.jl")

# model parameters
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

#  model = model_LDA(lattice, atoms, n_electrons=8)
model = model_atomic(lattice, atoms, n_electrons=8)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 5           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

tol = 1e-8
scfres = self_consistent_field(basis, tol=tol,
                               is_converged=DFTK.ScfConvergenceDensity(tol))

# newton algorithm
function newton(basis::PlaneWaveBasis{T}; ψ0=nothing,
                tol=1e-6, max_iter=100) where T

    ## setting parameters
    model = basis.model
    @assert model.spin_polarization in (:none, :spinless)
    @assert model.temperature == 0 # temperature is not yet supported
    filled_occ = DFTK.filled_occupation(model)
    n_bands = div(model.n_electrons, filled_occ)

    ## number of kpoints
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]
    ## number of eigenvalue/eigenvectors we are looking for
    N = n_bands

    ortho(ψk) = Matrix(qr(ψk).Q)
    if ψ0 === nothing
        ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), n_bands))
              for kpt in basis.kpoints]
    end

    ## vec and unpack
    # length of ψ0[ik]
    lengths = [length(ψ0[ik]) for ik = 1:Nk]
    starts = copy(lengths)
    starts[1] = 1
    for ik = 1:Nk-1
        starts[ik+1] = starts[ik] + lengths[ik]
    end
    pack(ψ) = vcat(Base.vec.(ψ)...) # TODO as an optimization, do that lazily? See LazyArrays
    unpack(ψ) = [@views reshape(ψ[starts[ik]:starts[ik]+lengths[ik]-1], size(ψ0[ik]))
                 for ik = 1:Nk]


    ## error list for convergence plots
    err_list = []
    err_ref_list = []
    k_list = []

    err = 1
    k = 0

    # orbitals to be updated along the iterations
    φ = deepcopy(ψ0)
    # this will also get updated along the iterations
    H = nothing
    ρ = nothing
    energies = nothing
    res = similar(ψ0)

    while err > tol && k < max_iter
        k += 1
        println("Iteration $(k)...")
        append!(k_list, k)

        # compute residual
        ρ = compute_density(basis, φ, occupation)
        energies, H = energy_hamiltonian(basis, φ, occupation; ρ=ρ[1])
        egval = [ zeros(Complex{T}, size(occupation[i])) for i = 1:length(occupation) ]

        for ik = 1:Nk
            φk = φ[ik]
            Hk = H.blocks[ik]
            egvalk = [φk[:,i]'*(Hk*φk[:,i]) for i = 1:N]
            rk = Hk*φk - hcat([egvalk[i] * φk[:,i] for i = 1:N]...)
            egval[ik] = egvalk
            res[ik] = rk
        end

        res = proj!(res, φ)

        # solve linear system with KrlyovKit
        function f(x)
            δφ = unpack(x)
            δφ = proj!(δφ, φ)
            ΩpKx = ΩplusK(basis, δφ, φ, ρ, H, egval, occupation)
            ΩpKx = proj!(ΩpKx, φ)
            pack(ΩpKx)
        end
        packed_proj!(ϕ,ψ) = proj!(unpack(ϕ), unpack(ψ))
        δφ_packed, info = linsolve(f, pack(res);
                                   tol=1e-15, verbosity=1,
                                   orth=OrthogonalizeAndProject(packed_proj!, pack(φ)))
        δφ = unpack(δφ_packed)
        δφ = proj!(δφ, φ)

        for ik = 1:Nk
            φk = φ[ik]
            δφk = δφ[ik]
            for i = 1:N
                φk[:,i] = φk[:,i] - δφk[:,i]
            end
            φk = ortho(φk)
            φ[ik] = φk
        end

        ρ_next = compute_density(basis, φ, occupation)
        err = norm(ρ_next[1].real - ρ[1].real)
        println(err)
        println(norm(res))
        append!(err_list, err)
        append!(err_ref_list, norm(ρ_next[1].real - scfres.ρ.real))

    end

    figure()
    semilogy(k_list, err_list, "x-", label="iter ρ")
    semilogy(k_list, err_ref_list, "x-", label="ref ρ")
end

ψ0 = deepcopy(scfres.ψ)
for ik = 1:length(ψ0)
    ψ0k = ψ0[ik]
    for i in 1:4
        ψ0k[:,i] += randn(size(ψ0k[:,i]))*1e-2
    end
end
φ0 = similar(ψ0)
for ik = 1:length(φ0)
    φ0[ik] = ψ0[ik][:,1:4]
    φ0[ik] = Matrix(qr(φ0[ik]).Q)
end
newton(basis; ψ0=φ0, tol=1e-12)
