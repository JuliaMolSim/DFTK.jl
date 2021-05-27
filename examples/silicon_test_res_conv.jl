# Very basic setup, useful for testing
using DFTK
using PyPlot
import Random

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

#  model = Model(lattice; atoms=atoms, n_electrons=2,
#                terms=[Kinetic(), AtomicLocal()])
model = model_LDA(lattice, atoms; n_electrons=2)
kgrid = [1,1,1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 20           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
basis_ref = PlaneWaveBasis(model, 10*Ecut; kgrid=kgrid)
tol = 1e-12

# compute the residual associated to a set of planewave φ, that is to say
# H(φ)*φ - λ.*φ where λ is the set of rayleigh coefficients associated to φ
# we also return the egval set for further computations
function compute_residual(basis::PlaneWaveBasis{T}, φ, occ) where T

    # necessary quantities
    Nk = length(basis.kpoints)
    ρ = compute_density(basis, φ, occ)
    energies, H = energy_hamiltonian(basis, φ, occ; ρ=ρ)

    # compute residual
    res = similar(φ)
    for ik = 1:Nk
        φk = φ[ik]
        N = size(φk, 2)
        Hk = H.blocks[ik]
        egvalk = [φk[:,i]'*(Hk*φk[:,i]) for i = 1:N]
        # compute residual at given kpoint as H(φ)φ - λφ
        res[ik] = Hk*φk - hcat([egvalk[i] * φk[:,i] for i = 1:N]...)
    end
    res
end

res_coarse = []
res_ref = []
res_Vφ = []
function residual_callback(info)

    # compute residual on coarse grid
    res_c = compute_residual(info.basis, info.ψ, info.occupation)

    # compute residual on reference grid
    ψr = DFTK.interpolate_blochwave(info.ψ, info.basis, basis_ref)
    res_r = compute_residual(basis_ref, ψr, info.occupation)

    # compute Vψ on the real space
    T = eltype(info.basis)
    kpt = info.ham.blocks[1].kpoint
    nband = size(info.ψ[1], 2)

    # storing quantities on basis
    Vψ_fourier = similar(info.ψ[1][:, 1])
    ψ_real = zeros(complex(T), info.basis.fft_size...)
    Vψ_real = zeros(complex(T), info.basis.fft_size...)
    Vlocψ_fourier_box = zeros(complex(T), info.basis.fft_size...)
    Vnonlocψ_fourier_box = zeros(complex(T), info.basis.fft_size...)

    # bigger basis for nonlocal potential
    bb = PlaneWaveBasis(model, 4*Ecut; kgrid=kgrid)
    ψ_bb = DFTK.interpolate_blochwave(info.ψ, info.basis, bb)
    ρ_bb = compute_density(bb, ψ_bb, info.occupation)
    energies, H_bb = energy_hamiltonian(bb, ψ_bb, info.occupation; ρ=ρ_bb)
    kpt_bb = H_bb.blocks[1].kpoint
    Vψ_fourier_bb = similar(ψ_bb[1][:, 1])
    ψ_real_bb = zeros(complex(T), bb.fft_size...)
    Vψ_real_bb = zeros(complex(T), bb.fft_size...)
    Vψ_fourier_box_bb = zeros(complex(T), bb.fft_size...)

    for iband = 1:nband

        # local potential
        Vψ_real .= 0
        Vψ_fourier .= 0
        G_to_r!(ψ_real, info.basis, kpt, info.ψ[1][:, iband])
        for op in info.ham.blocks[1].optimized_operators
            if op isa DFTK.RealSpaceMultiplication
                DFTK.apply!((fourier=Vψ_fourier, real=Vψ_real),
                            op,
                            (fourier=info.ψ[1][:, iband], real=ψ_real))
            end
        end
        r_to_G!(Vlocψ_fourier_box, info.basis, Vψ_real)

        # nonlocal potential, on a bigger grid
        Vψ_real_bb .= 0
        Vψ_fourier_bb .= 0
        for op in H_bb.blocks[1].optimized_operators
            if op isa DFTK.NonlocalOperator
                DFTK.apply!((fourier=Vψ_fourier_bb, real=Vψ_real_bb),
                            op,
                            (fourier=ψ_bb[1][:, iband], real=ψ_real_bb))
            end
        end
        Vψ_fourier_box_bb[kpt_bb.mapping] = Vψ_fourier_bb
        for (iG, G) in enumerate(DFTK.G_vectors(basis))
            iG_bb = DFTK.index_G_vectors(bb, G)
            Vnonlocψ_fourier_box[iG] += Vψ_fourier_box_bb[iG_bb]
        end

        # to fourier, in the box containing the Ecut sphere
        for (iG, G) in enumerate(DFTK.G_vectors(kpt))
            Vlocψ_fourier_box[kpt.mapping[iG]] = 0.0
            Vnonlocψ_fourier_box[kpt.mapping[iG]] = 0.0
        end
    end

    if info.stage == :finalize
        # plot residual convergence
        figure()
        semilogy(res_coarse, "x-", label="residual on coarse grid")
        semilogy(res_ref, "x-", label="residual on reference grid")
        semilogy(res_Vφ, "x-", label="Vψ")
        legend()
        xlabel("iterations")

        # plot residual components
        figure()
        Gs = [norm(G) for G in DFTK.G_vectors_cart(kpt)][:]
        Gs_ref = [norm(G) for G in DFTK.G_vectors_cart(basis_ref.kpoints[1])][:]
        Gs_box = [norm(G) for G in DFTK.G_vectors_cart(basis)][:]
        plot(Gs, res_c[1][:,1], "o", label="res coarse")
        plot(Gs_ref, res_r[1][:,1], "x", label="res ref")
        plot(Gs_box, vec(Vlocψ_fourier_box+Vnonlocψ_fourier_box), "+", label="Vψ")
        xlabel("|G|")
        legend()
    else
        append!(res_coarse, norm(res_c[1][:,1]))
        append!(res_ref, norm(res_r[1][:,1]))
        append!(res_Vφ, norm(Vlocψ_fourier_box+Vnonlocψ_fourier_box))
    end
    info
end

Random.seed!(1234)
filled_occ = DFTK.filled_occupation(model)
n_bands = div(model.n_electrons, filled_occ)
ortho(ψk) = Matrix(qr(ψk).Q)
ψ0 = [ortho(randn(ComplexF64, length(G_vectors(kpt)), n_bands))
      for kpt in basis.kpoints]
scfres = direct_minimization(basis, ψ0; tol=tol,
                             callback=info->residual_callback(info))
scfres.energies
