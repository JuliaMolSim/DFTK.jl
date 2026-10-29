# # Hybrid DFT band structures
#
# In this example we discuss how to compute hybrid DFT bands structures using DFTK.
#
# !!! warning "Preliminary implementation"
#     Hybrid-DFT in DFTK is still in its early stages. MPI parallelisation does not work
#     and all treatment of ``k``-point symmetry is disabled when using exact exchange terms.
#     We appreciate any issues, bug reports or PRs.
#
using AtomsBuilder
using DFTK
using PseudoPotentialData
using Wannier
using Brillouin
using CairoMakie
using Unitful
using UnitfulAtomic

pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf") 
system = bulk(:Si)

# ## SCF calculation
# We first perform a PBE calculation to get a good starting point. Since we will later Wannierise
# we make sure to converge 16 orbitals throughout the computational procedure.
model_PBE = model_DFT(system; pseudopotentials, functionals=PBE(), symmetries=false)
basis_PBE  = PlaneWaveBasis(model_PBE; Ecut=10, kgrid=[3, 3, 3])  # Not converged !

nbandsalg  = AdaptiveBands(model_PBE; n_bands_converge=16)
scfres_PBE = self_consistent_field(basis_PBE; tol=1e-4, nbandsalg)
nothing  # hide

# Next we converge an HSE calculation with the default number of bands, again in two steps.
# First using a more aggressive ACE compression techniques that is more memory efficient and
# slightly faster (by ignoring all unoccupied orbitals), but that gives the wrong unoccupied bands.
model  = model_DFT(system; pseudopotentials, functionals=HSE())
basis  = PlaneWaveBasis(model; basis_PBE.Ecut, basis_PBE.kgrid)

exxalg = AceExx(; sketch_with_extra_orbitals=false)
scfres = self_consistent_field(basis; tol=1e-6, scfres_PBE.ρ, scfres_PBE.ψ, scfres_PBE.occupation, exxalg,
                               scfres_PBE.eigenvalues, solver=ScfDampingSolver(), scfres_PBE.nbandsalg,
                               diagtolalg=AdaptiveDiagtol(; diagtol_max=1e-2, ratio_ρdiff=0.0005));
nothing  # hide

# Finally converge the HSE calculation with laxer ACE compression, that also takes
# the unoccupied orbitals into account (such that virtual bands are accurate)
exxalg = AceExx(; sketch_with_extra_orbitals=true)
scfres = self_consistent_field(basis; tol=1e-7, scfres.ρ, scfres.ψ, scfres.occupation, exxalg,
                               scfres.eigenvalues, scfres.solver, scfres.nbandsalg,
                               diagtolalg=AdaptiveDiagtol(; diagtol_max=1e-6, ratio_ρdiff=0.0005))
nothing  # hide

# ## Band structure computation
# For the PBE model we can directly use the [`compute_bands`](@ref) function of DFTK:
kpath = irrfbz_path(model)
bands_PBE = compute_bands(scfres_PBE, kpath; n_bands=12)

## Convert band data to format expected by Brillouin and to eV:
E_PBE = reduce(hcat, bands_PBE.eigenvalues)
E_PBE = ustrip.(auconvert.(u"eV", E_PBE))
E_PBE = E_PBE .- maximum(E_PBE[1:4, :])
nothing  # hide

# For HSE we proceed via wannierization. For this we setup SP3 projectors for silicon atoms:

projections = [DFTK.wannier_projectors_sp3(model.positions[1]);
               DFTK.wannier_projectors_sp3(model.positions[2])]

## Set the frozen window size. Since this is an insulator, about 2eV above the Fermi level
## should be good. Note that Wannier.jl uses eV units for energies:
dis_froz_max = ustrip(auconvert(u"eV", scfres.εF)) + 2.0

wannier_model = Wannier.Model(scfres;
                              n_wannier=length(projections),
                              n_bands=12,
                              projections,
                              dis_froz_max,
                              fileprefix="wannier/silicon")

## Run disentanglement, returns a (n_bands, n_wann, n_kpts) array 
U = disentangle(wannier_model; max_iter=500);

## InterpModel builds H(R) from model.U, so we store the U giving maximal localisation there
wannier_model.U .= U
nothing  # hide

# Based on the Wannier functions compute the bands via Wannier interpolation
interp_model = Wannier.InterpModel(wannier_model; kpath)
kinter = Brillouin.interpolate(kpath, density=40)  # same density as in DFTK by default
data = Wannier.interpolate(interp_model, kinter)
E_HSE = data .- maximum(data[1:4, :])
nothing  # hide

# ## Plot the bands
# Finally we plot the bands using the Makie interface of Brillouin:

fig = Figure(size=(700, 500))
CairoMakie.plot(fig[1, 1], kinter, permutedims(E_HSE), permutedims(E_PBE);
                label=["HSE", "PBE"],
                ylabel="E - E_VBM (eV)",
                ylims=(-13.0, 10.0),
                linewidth=2,
                linestyle=[:solid, :dash],
                color=[:royalblue, :black])

fig

# ## Further reading
# For more details on Wannierisation, see also [Wannierization using Wannier.jl or Wannier90](@ref).
