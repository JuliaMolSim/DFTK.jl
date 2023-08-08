# # Creating and modelling metallic supercells
#
# In this section we will be concerned with modelling supercells of aluminium.
# When dealing with periodic problems there is no unique definition of the
# lattice: Clearly any duplication of the lattice along an axis is also a valid
# repetitive unit to describe exactly the same system.
# This is exactly what a **supercell** is: An $n$-fold repetition along one of the
# axes of the original lattice.
#
# The following code achieves this for aluminium:

using DFTK
using LinearAlgebra
using ASEconvert

function aluminium_setup(repeat=1; Ecut=7.0, kgrid=[2, 2, 2])
    a = 7.65339
    lattice = a * Matrix(I, 3, 3)
    Al = ElementPsp(:Al; psp=load_psp("hgh/lda/al-q3"))
    atoms     = [Al, Al, Al, Al]
    positions = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    unit_cell = periodic_system(lattice, atoms, positions)

    ## Make supercell in ASE:
    ## We convert our lattice to the conventions used in ASE, make the supercell
    ## and then convert back ...
    supercell_ase = convert_ase(unit_cell) * pytuple((repeat, 1, 1))
    supercell     = pyconvert(AbstractSystem, supercell_ase)

    ## Unfortunately right now the conversion to ASE drops the pseudopotential information,
    ## so we need to reattach it:
    supercell = attach_psp(supercell; Al="hgh/lda/al-q3")

    ## Construct an LDA model and discretise
    ## Note: We disable symmetries explicitly here. Otherwise the problem sizes
    ##       we are able to run on the CI are too simple to observe the numerical
    ##       instabilities we want to trigger here.
    model = model_LDA(supercell; temperature=1e-3, symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid)
end;

# As part of the code we are using a routine inside the ASE,
# the [atomistic simulation environment](https://wiki.fysik.dtu.dk/ase/index.html)
# for creating the supercell and make use of the two-way interoperability of
# DFTK and ASE. For more details on this aspect see the documentation
# on [Input and output formats](@ref).

# Write an example supercell structure to a file to plot it:
setup = aluminium_setup(5)
convert_ase(periodic_system(setup.model)).write("al_supercell.png")

#md # ```@raw html
#md # <img src="../al_supercell.png" width=500 height=500 />
#md # ```
#nb # <img src="https://docs.dftk.org/stable/examples/al_supercell.png" width=500 height=500 />

# As we will see in this notebook the modelling of a system generally becomes
# harder if the system becomes larger.
#
# - This sounds like a trivial statement as *per se* the cost per SCF step increases
#   as the system (and thus $N$) gets larger.
# - But there is more to it:
#   If one is not careful also the *number of SCF iterations* increases
#   as the system gets larger.
# - The aim of a proper computational treatment of such supercells is therefore
#   to ensure that the **number of SCF iterations remains constant** when the
#   system size increases.

# For achieving the latter DFTK by default employs the `LdosMixing`
# preconditioner [^HL2021] during the SCF iterations. This mixing approach is
# completely parameter free, but still automatically adapts to the treated
# system in order to efficiently prevent charge sloshing. As a result,
# modelling aluminium slabs indeed takes roughly the same number of SCF iterations
# irrespective of the supercell size:
#
# [^HL2021]:
#    M. F. Herbst and A. Levitt.
#    *Black-box inhomogeneous preconditioning for self-consistent field iterations in density functional theory.*
#    J. Phys. Cond. Matt *33* 085503 (2021). [ArXiv:2009.01665](https://arxiv.org/abs/2009.01665)
#

self_consistent_field(aluminium_setup(1); tol=1e-4);

#-

self_consistent_field(aluminium_setup(2); tol=1e-4);

#-

self_consistent_field(aluminium_setup(4); tol=1e-4);

# When switching off explicitly the `LdosMixing`, by selecting `mixing=SimpleMixing()`,
# the performance of number of required SCF steps starts to increase as we increase
# the size of the modelled problem:

self_consistent_field(aluminium_setup(1); tol=1e-4, mixing=SimpleMixing());

#-

self_consistent_field(aluminium_setup(4); tol=1e-4, mixing=SimpleMixing());

# For completion let us note that the more traditional `mixing=KerkerMixing()`
# approach would also help in this particular setting to obtain a constant
# number of SCF iterations for an increasing system size (try it!). In contrast
# to `LdosMixing`, however, `KerkerMixing` is only suitable to model bulk metallic
# system (like the case we are considering here). When modelling metallic surfaces
# or mixtures of metals and insulators, `KerkerMixing` fails, while `LdosMixing`
# still works well. See the [Modelling a gallium arsenide surface](@ref) example
# or [^HL2021] for details. Due to the general applicability of `LdosMixing` this
# method is the default mixing approach in DFTK.
