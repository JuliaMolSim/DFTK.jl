# # [Pseudopotentials](@id pseudopotentials)
#
# In this example, we'll look at how to use various pseudopotential (PSP)
# formats in DFTK and discuss briefly the utility and importance of
# pseudopotentials.
#
# Currently, DFTK supports norm-conserving (NC) PSPs in
# separable (Kleinman-Bylander) form. Two file formats can currently
# be read and used: analytical Hartwigsen-Goedecker-Hutter (HGH) PSPs
# and numeric Unified Pseudopotential Format (UPF) PSPs.
#
# In brief, the pseudopotential approach replaces the all-electron
# atomic potential with an effective atomic potential. In this pseudopotential,
# tightly-bound core electrons are completely eliminated ("frozen") and
# chemically-active valence electron wavefunctions are replaced with
# smooth pseudo-wavefunctions whose Fourier representations decay quickly.
# Both these transformations aim at reducing the number of Fourier modes required
# to accurately represent the wavefunction of the system, greatly increasing
# computational efficiency.
#
# Different PSP generation codes produce various file formats which contain the
# same general quantities required for pesudopotential evaluation. HGH PSPs
# are constructed from a fixed functional form based on Gaussians, and the files
# simply tablulate various coefficients fitted for a given element. UPF PSPs
# take a more flexible approach where the functional form used to generate the
# PSP is arbitrary, and the resulting functions are tabulated on a radial grid
# in the file. The UPF file format is documented here:
# http://pseudopotentials.quantum-espresso.org/home/unified-pseudopotential-format.
#
# In this example, we will compare the convergence of an analytical HGH PSP with
# a modern UPF PSP from PseudoDojo (http://www.pseudo-dojo.org/).
# Then, we will compare the bandstructure at the converged parameters calculated
# using the two PSPs.

using DFTK
using Downloads
using Unitful
using UnitfulAtomic
using Plots

# Here, we will use Perdew-Wang LDA PSP from PseudoDojo

URL_UPF = "https://raw.githubusercontent.com/JuliaMolSim/PseudoLibrary/main/pseudos/pd_nc_sr_lda_standard_04_upf/Li.upf"

# We load the HGH and UPF PSPs using `load_psp`, which determines the
# file format using the file extension.

psp_hgh = load_psp("hgh/lda/li-q3.hgh");
path_upf = Downloads.download(URL_UPF, joinpath(tempdir(), "Li.upf"))
psp_upf = load_psp(path_upf);

# First, we'll take a look at the energy cutoff convergence of these two pseudopotentials.
# For both pseudos, a reference energy is calculated with a cutoff of 140 Hartree, and
# SCF calculations are run at increasing cutoffs until 1 meV / atom convergence is reached.

#md # ```@raw html
#md # <img src="../../assets/li_pseudos_ecut_convergence.png" width=600 height=400 />
#md # ```
#nb # <img src="https://docs.dftk.org/stable/assets/li_pseudos_ecut_convergence.png" width=600 height=400 />

# The converged cutoffs are 128 Ha and 36 Ha for the HGH
# and UPF pseudos respectively. We see that the HGH pseudopotential
# is much *harder*, i.e. it requires a higher energy cutoff, than the UPF PSP. In general,
# numeric pseudopotentials tend to be softer than analytical pseudos because of the
# flexibility of sampling arbitrary functions on a grid.

# Next, to see that the different pseudopotentials give reasonbly similar results,
# we'll look at the bandstructures calculated using the HGH and UPF PSPs. Even though
# the convered cutoffs are 128 and 36 Ha, we perform these calculations with a cutoff of
# 24 Ha for both PSPs.

function run_bands(psp)
    a = -1.5387691950u"Å"
    b = -2.6652264269u"Å"
    c = -4.9229470000u"Å"
    lattice = [
        [ a a  0];
        [-b b  0];
        [ 0 0 -c]
    ]
    Li = ElementPsp(:Li, psp=psp)
    atoms     = [Li, Li]
    positions = [[1/3, 2/3, 1/4],
                 [2/3, 1/3, 3/4]]

    # These are (as you saw above) completely unconverged parameters
    model = model_LDA(lattice, atoms, positions; temperature=1e-2)
    basis = PlaneWaveBasis(model; Ecut=24, kgrid=[6, 6, 4])
    
    scfres = self_consistent_field(basis, tol=1e-6)
    bandplot = plot_bandstructure(scfres)
    
    return (basis=basis, scfres=scfres, bandplot=bandplot)
end

# The SCF and bandstructure calculations can then be performed using
# the two PSPs ...

result_hgh = run_bands(psp_hgh);
result_upf = run_bands(psp_upf);

# ... and the respective bandstructures are plotted:

plot(result_hgh.bandplot, result_upf.bandplot, titles=["HGH" "UPF"], size=(800,400))