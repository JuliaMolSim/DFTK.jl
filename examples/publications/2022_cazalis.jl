# Model of graphene confined to 2 spatial dimensions studied
# in the paper by Cazalis (arxiv, 2022, TODO add ref)
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

    ρtot_fourier = r_to_G(basis, total_density(ρ))
    pot_fourier = poisson_green_coeffs .* ρtot_fourier
    pot_real = G_to_r(basis, pot_fourier)
    E = real(dot(pot_fourier, ρtot_fourier) / 2)
    ops = [DFTK.RealSpaceMultiplication(basis, kpt, pot_real) for kpt in basis.kpoints]
    (E=E, ops=ops)
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
scfres = self_consistent_field(basis, tol=1e-10)

## Choose the points of the band diagram, in reduced coordinates (in the (b1,b2) basis)
Γ  = [0, 0, 0]
K  = [ 1, 1, 0]/3
Kp = [-1, 2, 0]/3
M  = (K + Kp)/2
kpath_coords = [Γ, K, M, Γ]
kpath_names  = ["Γ", "K", "M", "Γ"]

## Build the path
kline_density = 20
function build_path(k1, k2)
    target_Δk = 1/kline_density  # the actual Δk is |k2-k1|/npt
    npt = ceil(Int, norm(model.recip_lattice * (k2-k1)) / target_Δk)
    [k1 + t * (k2-k1) for t in range(0, 1, length=npt)]
end
kcoords = []
for i = 1:length(kpath_coords)-1
    append!(kcoords, build_path(kpath_coords[i], kpath_coords[i+1]))
end
klabels = Dict(kpath_names[i] => kpath_coords[i] for i=1:length(kpath_coords))

## Plot the bands
band_data = compute_bands(basis, kcoords; n_bands=5, scfres.ρ)
p = DFTK.plot_band_data(band_data; klabels, markersize=nothing)
Plots.hline!(p, [scfres.εF], label="", color="black")
Plots.ylims!(p, -Inf,Inf)
p
