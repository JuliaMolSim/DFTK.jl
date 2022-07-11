# # Graphene band structure

# This example plots the band structure of graphene, a 2D material. 2D band structures are not supported natively (yet), so we manually build a custom path in reciprocal space.

using DFTK
using Unitful
using UnitfulAtomic

# Define the convergence parameters (these should be increased in production)
L = 20 # height of the simulation box
kgrid = [6, 6, 1]
Ecut = 15
temperature = 1e-3

# Define the geometry and pseudopotential
a = 4.66 # lattice constant
a1 = a*[1/2,-sqrt(3)/2, 0]
a2 = a*[1/2, sqrt(3)/2, 0]
a3 = L*[0  , 0        , 1]
lattice = [a1 a2 a3]
C1 = [1/3,-1/3,0.0] # in reduced coordinates
C2 = -C1
positions = [C1, C2]
C = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4"))
atoms = [C, C]

# Run SCF
model = model_PBE(lattice, atoms, positions; temperature)
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres = self_consistent_field(basis)

# Choose the points of the band diagram, in reduced coordinates (in the (b1,b2) basis)
Γ = [0, 0, 0]
K1 = [ 1, 1, 0]/3
K2 = [-1, 2, 0]/3
M = (K1 + K2)/2
kpath_coords = [Γ, K1, M, Γ]
kpath_names = ["Γ", "K", "M", "Γ"]

# Build the path
kline_density = 20
function build_path(k1, k2)
    target_Δk = 1/kline_density # the actual Δk is |k2-k1|/npt
    npt = ceil(Int, norm(model.recip_lattice * (k2-k1)) / target_Δk)
    [k1 + t * (k2-k1) for t in range(0, 1, length=npt)]
end
kcoords = []
for i = 1:length(kpath_coords)-1
    append!(kcoords, build_path(kpath_coords[i], kpath_coords[i+1]))
end
klabels = Dict(kpath_names[i] => kpath_coords[i] for i=1:length(kpath_coords))

# Plot the bands
band_data = compute_bands(basis, kcoords; scfres.ρ)
display(DFTK.plot_band_data(band_data; scfres.εF, klabels, units=u"hartree"))
