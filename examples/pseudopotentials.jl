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
using LinearAlgebra
using Unitful
using UnitfulAtomic
using Plots
using GZip

# Here, we will use Perdew-Wang LDA PSP from PseudoDojo
URL_UPF = "http://www.pseudo-dojo.org/pseudos/nc-sr-04_pw_standard/Li.upf.gz"

function download_gunzip_upf(url)
    path_gz = download(url, joinpath(tempdir(), "psp.upf.gz"))
    text_upf = GZip.open(path_gz, "r") do io
        read(io, String)
    end
    path_upf = joinpath(tempdir(), "psp.upf")
    open(path_upf, "w") do io
        write(io, text_upf)
    end
    path_upf
end

function run_scf(Ecut, psp, tol)
    println("Ecut = $Ecut")
    println("----------------------------------------------------")
    a = 5.0
    lattice   = a * Matrix(I, 3, 3)
    atoms     = [ElementPsp(:Li, psp=psp)]
    positions = [zeros(3)]

    model = model_LDA(lattice, atoms, positions; temperature=1e-2)
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=[8, 8, 8])
    self_consistent_field(basis; tol)
end

function converge_Ecut(Ecuts, psp, tol)
    energies = [run_scf(Ecut, psp, tol/100).energies.total for Ecut in Ecuts]
    errors = abs.(energies[begin:end-1] .- energies[end])
    iconv = findfirst(errors .< tol)
    (Ecuts=Ecuts[begin:end-1], errors, Ecut_conv=Ecuts[iconv])
end

# We load the HGH and UPF PSPs using `load_psp`, which determines the
# file format using the file extension.

psp_hgh = load_psp("hgh/lda/li-q3.hgh");
psp_upf = load_psp(download_gunzip_upf(URL_UPF));

# Next, we define some parameters for the energy cutoff convergence.
# These are fairly lose parameters; in practice tighter parameters
# and higher cutoffs might be needed. The PseudoDojo provides recommended
# cutoffs for its pseudopotentials as notes on the periodic table.
# Ecuts = 28:4:64
Ecuts = 20:4:68
tol   = 1e-2

conv_hgh = converge_Ecut(Ecuts, psp_hgh, tol)
conv_upf = converge_Ecut(Ecuts, psp_upf, tol)

# From this small convergence study, we can look at the difference in convergence w.r.t.
# `Ecut` between the analytical and numeric pseudos. We see that the HGH pseudopotential
# is much *harder*, i.e. it requires a higher energy cutoff, than the UPF PSP. In general,
# numeric pseudopotentials tend to be softer than analytical pseudos because of the
# flexibility of sampling an arbitrary functions on a grid. 

begin
    plt = plot(xlabel="Ecut [Ha]", ylabel="Error [Ha]")
    plot!(plt, conv_hgh.Ecuts, conv_hgh.errors, label="HGH")
    plot!(plt, conv_upf.Ecuts, conv_upf.errors, label="PseudoDojo UPF")
    hline!(plt, [tol], label="tol", color=:grey, linestyle=:dash)
end

#md # ```@raw html
#md # <img src="../../assets/Li_convergence.png" width=600 height=400 />
#md # ```
#nb # <img src="https://docs.dftk.org/stable/assets/Li_convergence.png" width=600 height=400 />

# Next, we can look at bandstructures calculated using the converged parameters we've just
# found. Because both potentials are converged to the same extent, we expect the
# bands diagrams to look quite similar. First, we'll define a function to run the SCF
# and bands calculations.

function run_bands(Ecut, psp, tol)
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

    model = model_LDA(lattice, atoms, positions; temperature=1e-2)
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=[8, 8, 5])
    
    scfres = self_consistent_field(basis, tol=tol/100)
    bandplot = plot_bandstructure(scfres)
    
    return (basis=basis, scfres=scfres, bandplot=bandplot)
end

# The SCF and bandstructure calculations can then be performed using
# the two PSPs ...
 
result_hgh = run_bands(conv_hgh.Ecut_conv, psp_hgh, tol);
result_upf = run_bands(conv_upf.Ecut_conv, psp_upf, tol);

# ... and the respective bandstructures are plotted:

plot(result_hgh.bandplot, result_upf.bandplot, titles=["HGH" "UPF"], size=(800,400))

#md # ```@raw html
#md # <img src="../../assets/Li_bandstructure.png" width=600 height=400 />
#md # ```
#nb # <img src="https://docs.dftk.org/stable/assets/Li_bandstructure.png" width=600 height=400 />