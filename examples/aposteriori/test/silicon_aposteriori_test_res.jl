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
using DataFrames
using GLM

# import aux file
include("../aposteriori_tools.jl")
include("../aposteriori_callback.jl")
#  include("newton.jl")

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
Ecut_ref = 5       # kinetic energy cutoff in Hartree
samp = 2           # supersampling parameter

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

    Ecut = 25*Ecut_ref
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, supersampling=samp)

    ### ploting fourier coefficients of different quantities
    figure()
    title("residual component in basis for Ecut = $(Ecut), Ecut_ref = $(Ecut_ref)")
    kpt = basis.kpoints[1]
    kpt_ref = basis_ref.kpoints[1]
    G_energies = [sum(abs2.(G + kpt.coordinate_cart)) ./ 2 for G in DFTK.G_vectors_cart(kpt)][:]
    # residual
    φ = DFTK.interpolate_blochwave(φ_ref, basis_ref, basis)
    #  ii = 5
    #  φ[1][ii+1:end,1] .= 0
    #  φ[1][:,1] ./= norm(φ[1][:,1])
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
    Vloc_fourier_grid = r_to_G(basis, Vloc)

    #  # compute convolution
    #  Vφc = zeros(ComplexF64, length(φ[1][:,1]))
    #  for (iG, G) in enumerate(G_vectors(kpt))
    #      #  println((iG, length(G_vectors(kpt))))
    #      for (iGp, Gp) in enumerate(G_vectors(kpt_ref))
    #          ΔG = G-Gp
    #          ΔiG = DFTK.index_G_vectors(basis, ΔG)
    #          Vφc[iG] += Vloc_fourier_grid[ΔiG] * φ[1][iGp,1] / sqrt(model.unit_cell_volume)
    #      end
    #  end
    #  semilogy(G_energies, abs.(Vφc), "x", label="Vφ convol")

    semilogy(G_energies, abs.(Vloc_fourier), "*", label="Vloc")

    #  testing slope
    r = Si.psp.rloc^2
    nz_G_energies = []
    nz_G_energies_dec = []
    nz_Vloc = []
    GG = model.recip_lattice*G_vectors(kpt_ref)[2]
    G_energies_dec = [sum(abs2.(G - GG + kpt.coordinate_cart)) ./ 2 for G in DFTK.G_vectors_cart(kpt)][:]
    for i in 1:length(G_energies)
        if abs(Vloc_fourier[i]) > 1e-10 && G_energies[i] > 50
            append!(nz_Vloc, abs(Vloc_fourier[i]))
            append!(nz_G_energies, G_energies[i])
            append!(nz_G_energies_dec, G_energies_dec[i])
        end
    end
    data = DataFrame(X=Float64.(nz_G_energies), Y=Float64.(log.(nz_Vloc)))
    ols = lm(@formula(Y ~ X), data)
    slope = coef(ols)[2]
    semilogy(nz_G_energies, exp.(coef(ols)[2] .* nz_G_energies .+ coef(ols)[1]), label="linreg")
    semilogy(nz_G_energies, exp.(- r .* nz_G_energies .+ coef(ols)[2]), label="rloc^2")
    #  semilogy(nz_G_energies,
    #           0.05 .* (abs(φ[1][1,1]) * exp.(- r .* nz_G_energies )
    #                   .+ abs(φ[1][2,1]) * exp.(- r .* nz_G_energies_dec)), "*", label="test")
    println(ols)
    println(r)
    println(slope)

    legend()
    xlabel("Energy mode")
    ylabel("residual")
end
