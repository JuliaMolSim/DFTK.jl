using DFTK
using Plots

# Calculation parameters
kgrid = [4, 4, 4]       # k-Point grid
supercell = [1, 1, 1]   # Lattice supercell
Ecut = 15               # kinetic energy cutoff in Hartree
n_bands = 8             # number of bands to plot in the bandstructure

# Setup silicon lattice
a = 10.263141334305942  # Silicon lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# Make a supercell if desired
pystruct = pymatgen_structure(lattice, atoms)
pystruct.make_supercell(supercell)
lattice = load_lattice(pystruct)
atoms = [Si => [s.frac_coords for s in pystruct.sites]]

model = model_LDA(lattice, atoms)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)


n_tot_matvec = 0
function scf_lobpcg_diagnostics(info)
    global n_tot_matvec
    DFTK.scf_default_callback(info)
    println("    $(info.diagonalisation.n_matvec)  matvecs")
    min_resid = minimum(minimum(info.diagonalisation.residual_norms))
    max_resid = maximum(maximum(info.diagonalisation.residual_norms))
    println("    [$min_resid, $max_resid]")
    println("    $(info.diagonalisation.iterations)")
    n_tot_matvec += info.diagonalisation.n_matvec
end

# profile = :abinit
# profile = :toldep
# profile = :tolnext
profile = :old
println("Using SCF profile: $profile")
println()

# Run SCF. Note Silicon is a semiconductor, so we use an insulator
# occupation scheme. This will cause warnings in some models, because
# e.g. in the :reduced_hf model silicon is a metal
# scfres = direct_minimization(basis)
scfres = self_consistent_field(basis, tol=1e-10, callback=scf_lobpcg_diagnostics,
                               profile=profile)
println()
println("total matvecs: $n_tot_matvec")
println()

STOP

# Print energies and plot bands
println()
display(scfres.energies)
gui(plot_bandstructure(scfres, n_bands))
