# # Density Guesses
#
# We compare four different approaches to charge density initialization for starting a
# density-based SCF.

# First we set up our problem
using DFTK
using LinearAlgebra
using LazyArtifacts
import Main: @artifact_str # hide

# We use a numeric norm-conserving PSP in UPF format from the
# [PseudoDojo](http://www.pseudo-dojo.org/) v0.4 scalar-relativistic LDA standard stringency
# family because it contains valence charge density which can be used for a more tailored
# density guess.
UPF_PSEUDO = artifact"pd_nc_sr_lda_standard_0.4.1_upf/Si.upf"

function silicon_scf(guess_method)
    a = 10.26  # Silicon lattice constant in Bohr
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si; psp=load_psp(UPF_PSEUDO))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]

    model = model_LDA(lattice, atoms, positions)
    basis = PlaneWaveBasis(model; Ecut=12, kgrid=[4, 4, 4])

    ρguess = guess_density(basis, guess_method)

    is_converged = DFTK.ScfConvergenceEnergy(1e-10)
    self_consistent_field(basis; is_converged, ρ=ρguess)
end;

# ## Random guess
# The random density is normalized to the number of electrons provided.
scfres_random = silicon_scf(RandomDensity());

# ## Superposition of Gaussian densities
# The Gaussians are defined by a tabulated atom decay length.
scfres_gaussian = silicon_scf(ValenceGaussianDensity());

# ## Automatic density guess
# This method will automatically use valence charge densities from from pseudopotentials
# that provide them and Gaussian densities for elements which do not have pseudopotentials
# or whose pseudopotentials don't provide valence charge densities. To force all elements
# to use valence charge densities (and error where any element doesn't have them), use
# `PspDensityGuess()`. 
scfres_psp = silicon_scf(ValenceAutoDensity());
