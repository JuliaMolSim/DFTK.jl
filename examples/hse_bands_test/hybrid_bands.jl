# # Hybrid DFT band structures
#
# This is a quick sketch how to compute Hybrid DFT band structures using DFTK.
#
using AtomsBuilder
using DFTK
using PseudoPotentialData
using Wannier
using Brillouin
using GLMakie
using Unitful
using UnitfulAtomic

pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf") 
system = bulk(:Si)

# First perform a PBE calculation to get a good starting point.
model  = model_DFT(system; pseudopotentials, functionals=PBE(), symmetries=false)
basis  = PlaneWaveBasis(model; Ecut=18, kgrid=[3, 3, 3])

n_bands    = 16
nbandsalg  = AdaptiveBands(model; n_bands_converge=n_bands)
scfres_PBE = self_consistent_field(basis; nbandsalg, tol=1e-4)
nothing  # hide

DFTK.reset_timer!(DFTK.timer)

model  = model_DFT(system; pseudopotentials, functionals=HSE())
basis  = PlaneWaveBasis(model; basis.Ecut, basis.kgrid)
scfres = self_consistent_field(basis; tol=1e-7, scfres_PBE.ρ, scfres_PBE.ψ, scfres_PBE.occupation,
                               scfres_PBE.eigenvalues, solver=ScfDampingSolver(),
                               nbandsalg, diagtolalg=AdaptiveDiagtol(; diagtol_max=1e-2, ratio_ρdiff=0.0005));

println()
println(DFTK.timer)
println()

scfres = self_consistent_field(basis; tol=1e-7, scfres.ρ, scfres.ψ, scfres.occupation,
                               scfres.eigenvalues, solver=ScfDampingSolver(),
                               nbandsalg, exxalg=VanillaExx(),
                               diagtolalg=AdaptiveDiagtol(; diagtol_max=1e-7));
εF_eV = scfres.εF * 27.211386245988   # Hartree → eV
nothing  # hide

# See also [`HSE`](@ref) and [`model_HF`](@ref) for setting up Hartree-Fock calculations.

# =============================================================
# Clauded Wannier setup
# =============================================================

"""
sp3-like hybrid initial guess: (s + √3 d̂·p) / 2 on the atom at `center`
(fractional coordinates), pointing along the Cartesian direction `direction`.
Built as a linear combination of DFTK's hydrogenic projections, so it can be
passed to `Wannier.Model(scfres; projections=...)` like any other projection.
"""
struct SP3WannierProjection
    center::Vector{Float64}
    direction::Vector{Float64}
    n::Int        # radial function index (as `r` in Wannier90); n = 1 is nodeless
    α::Float64    # Z/a in 1/Bohr
end
 
function (proj::SP3WannierProjection)(basis, ps)
    d = proj.direction / norm(proj.direction)
    orb(l, m) = DFTK.HydrogenicWannierProjection(proj.center, proj.n, l, m, proj.α)(basis, ps)
    # DFTK real harmonics: (l=1, m=1) = p_x, (1,-1) = p_y, (1,0) = p_z
    s, px, py, pz = orb(0, 0), orb(1, 1), orb(1, -1), orb(1, 0)
    (s .+ sqrt(3) .* (d[1] .* px .+ d[2] .* py .+ d[3] .* pz)) ./ 2
end
 
"""Cartesian vectors from atom `iatom` to its `n_neigh` nearest neighbours."""
function bond_directions(lattice, positions, iatom; n_neigh=4)
    ri = lattice * positions[iatom]
    candidates = [lattice * (positions[j] + [n1, n2, n3]) - ri
                  for j in eachindex(positions), n1 in -1:1, n2 in -1:1, n3 in -1:1]
    candidates = sort(filter(v -> norm(v) > 1e-8, vec(candidates)); by=norm)
    candidates[1:n_neigh]
end
 
n_wannier = 8
α = 1 / 1.8897261   # 1 Å⁻¹ in 1/Bohr, the Wannier90 default; only shapes the guess
projections = [SP3WannierProjection(model.positions[iatom], d, 1, α)
               for iatom in eachindex(model.positions)
               for d in bond_directions(model.lattice, model.positions, iatom)]
@assert length(projections) == n_wannier
 
dis_froz_max = εF_eV + 2.0   # eV: freeze valence + bottom of conduction band

# =============================================================

# TODO: This setup will need a bit of tuning ... see the Wannier.jl docs
wannier_model = Wannier.Model(scfres;
                              n_wannier,
                              n_bands=12,
                              projections,
                              dis_froz_max,
                              fileprefix="wannier/silicon")

U = disentangle(wannier_model; max_iter=500)   # (n_bands, n_wann, n_kpts) array
show(stdout, MIME("text/plain"), omega(wannier_model, U)); println()

# InterpModel builds H(R) from model.U, so store the optimised gauge there
wannier_model.U .= U

kline_density=40u"bohr"
kpath  = irrfbz_path(model)
kinter = Brillouin.interpolate(kpath, density=austrip(kline_density))

interp_model = Wannier.InterpModel(wannier_model; kpath)
data = Wannier.interpolate(interp_model, kinter)
E_HSE = data .- maximum(data[1:4, :])

bands_PBE = compute_bands(scfres_PBE, ExplicitKpoints(kinter); n_bands=n_wannier)
E_PBE = reduce(hcat, bands_PBE.eigenvalues) .- bands_PBE.εF
E_PBE = ustrip.(auconvert.(u"eV", E_PBE))
E_PBE = E_PBE .- maximum(E_PBE[1:4, :])

fig = GLMakie.Figure(size=(700, 500))
GLMakie.plot(fig[1, 1], kinter, permutedims(E_HSE), permutedims(E_PBE);
             label=["HSE", "PBE"],
             ylabel    = "E - E_VBM (eV)",
             ylims     = (-13.0, 10.0),
             linewidth = 2,
             linestyle = [:solid, :dash],
             color     = [:royalblue, :black])

GLMakie.save("/tmp/si_hybrid_bands.png", fig)
# GLMakie.display(fig)
fig
