# Model of graphene confined to 2 spatial dimensions studied
# in the paper by Cazalis (arxiv, 2022, https://arxiv.org/abs/2207.09893)
# The pure 3D Coulomb 1/|x| interaction is used, without pseudopotential.

using DFTK
using Plots

## redefine Hartree interaction
struct Hartree2D end
struct Term2DHartree <: DFTK.TermNonlinear end
(t::Hartree2D)(basis) = Term2DHartree()
function DFTK.ene_ops(term::Term2DHartree, basis::PlaneWaveBasis{T},
                      ψ, occ; ρ, kwargs...) where {T}
    ## 2D Fourier transform of 3D Coulomb interaction 1/|x|
    poisson_green_coeffs = 2T(π) ./ [norm(G) for G in G_vectors_cart(basis)]
    poisson_green_coeffs[1] = 0  # DC component

    ρtot_fourier = fft(basis, total_density(ρ))
    pot_fourier = poisson_green_coeffs .* ρtot_fourier
    pot_real = irfft(basis, pot_fourier)
    E = real(dot(pot_fourier, ρtot_fourier) / 2)
    ops = [DFTK.RealSpaceMultiplication(basis, kpt, pot_real) for kpt in basis.kpoints]
    (; E, ops)
end

## define electron-nuclei interaction
struct Element2DCoulomb <: DFTK.Element end
function DFTK.local_potential_fourier(el::Element2DCoulomb, q::Real)
    ## = ∫ V(r) exp(-ix⋅q) dx
    if q == 0
        zero(q)
    else
        -2π/q
    end
end

L = 1.0
kgrid = [10, 10, 1]  # increase this for production
Ecut = 500  # increase this for production
height = 0

## Define the geometry and pseudopotential
a1 = L * [1/2, -sqrt(3)/2, 0]
a2 = L * [1/2,  sqrt(3)/2, 0]
a3 = height * [0, 0, 1]
lattice = [a1 a2 a3]
C1 = [1/3, -1/3, 0.0]  # in reduced coordinates
C2 = -C1
positions = [C1, C2]

c = Element2DCoulomb()
atoms = [c, c]

model = Model(lattice, atoms, positions; temperature=1e-4, smearing=Smearing.Gaussian(),
              terms=[Kinetic(), Hartree2D(), AtomicLocal()],
              n_electrons=2)
basis = PlaneWaveBasis(model; Ecut, kgrid)

## Run SCF
scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceEnergy(1e-10))

## Plot bands
kpath = irrfbz_path(model; dim=2, space_group_number=13)
p = plot_bandstructure(scfres, kpath; n_bands=5)
Plots.hline!(p, [scfres.εF], label="", color="black")
Plots.ylims!(p, (-Inf, Inf))
p
