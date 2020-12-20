# This file contains test of the estimator
#
# P-P* = (Ω+K)^{-1}[P,[P,H(P)]]
#
# when the error we look at is the basis error : P* is computed for a reference
# Ecut_ref and then we measure the error P-P* and the residual obtained for
# smaller Ecut (currently, only Nk = 1 kpt only is supported)
#
#            !!! NOT OPTIMIZED YET, WE USE PLAIN DENSITY MATRICES !!!
#

# Very basic setup, useful for testing
using DFTK
using LinearAlgebra
using PyPlot

# import aux file
include("aposteriori_operators.jl")
include("aposteriori_callback.jl")
include("newton.jl")

# model parameters
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# define different models
#  modelLDA = model_LDA(lattice, atoms)
#  modelHartree = model_atomic(lattice, atoms; extra_terms=[Hartree()])
#  modelAtomic = model_atomic(lattice, atoms, n_electrons=2)
modelAtomic = Model(lattice; atoms=atoms,
                    terms=[Kinetic(), AtomicLocal()],
                    n_electrons=2)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-10
tol_krylov = 1e-15
Ecut_ref = 50           # kinetic energy cutoff in Hartree

for model in [modelAtomic]#, modelHartree, modelLDA]
    println("--------------------------------")
    println("--------------------------------")
    ## reference density matrix
    println("--------------------------------")
    println("reference computation")
    basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)
    scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                       determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                       is_converged=DFTK.ScfConvergenceDensity(tol))
    T = typeof(scfres_ref.ρ.real[1])
    ## number of kpoints
    Nk = length(basis_ref.kpoints)
    ## number of eigenvalue/eigenvectors we are looking for
    filled_occ = DFTK.filled_occupation(model)
    N = div(model.n_electrons, filled_occ)
    occupation = [filled_occ * ones(T, N) for ik = 1:Nk]
    φ_ref = similar(scfres_ref.ψ)
    for ik = 1:Nk
        φ_ref[ik] = scfres_ref.ψ[ik][:,1:N]
    end
    gap = scfres_ref.eigenvalues[1][N+1] - scfres_ref.eigenvalues[1][N]
    H_ref = scfres_ref.ham
    println(typeof.(H_ref.blocks[1].operators))

    Ecut = 3*Ecut_ref
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

    figure()
    title("residual component in basis for Ecut = $(Ecut), Ecut_ref = $(Ecut_ref)")
    kpt = basis.kpoints[1]
    G_energies = [sum(abs2.(G + kpt.coordinate_cart)) ./ 2 for G in DFTK.G_vectors_cart(kpt)][:]
    # residual
    φ = DFTK.interpolate_blochwave(φ_ref, basis_ref, basis)
    ρ = compute_density(basis, φ, occupation)
    energies, H = energy_hamiltonian(basis, φ, occupation; ρ=ρ[1])
    res = compute_residual(basis, φ, occupation)
    semilogy(G_energies, abs.(res[1][:,1]), "^", label="res")
    # wavefunction
    semilogy(G_energies, abs.(φ[1][:,1]), "o", label="φ")
    # Vφ
    T = eltype(basis)
    scratch = (
               ψ_reals=[zeros(complex(T), basis.fft_size...) for tid = 1:Threads.nthreads()],
               Hψ_reals=[zeros(complex(T), basis.fft_size...) for tid = 1:Threads.nthreads()]
              )
    ops_no_kin = [op for op in H.blocks[1].operators
    if !(op isa DFTK.FourierMultiplication)]
    H_no_kin = HamiltonianBlock(basis, kpt, ops_no_kin, scratch)
    Vφ = mul!(similar(φ[1]), H_no_kin, φ[1])
    semilogy(G_energies, abs.(Vφ[:,1]), "+", label="Vφ")
    # V
    Vloc = Complex{Float64}.(DFTK.total_local_potential(H))
    Vloc_fourier = r_to_G(basis, kpt, Vloc)
    semilogy(G_energies, abs.(Vloc_fourier), "*", label="Vloc")
    legend()
    xlabel("Energy mode")
    ylabel("residual")
end
