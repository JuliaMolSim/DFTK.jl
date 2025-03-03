# # Using DFTK on GPUs
#
# In this example we will look how DFTK can be used on
# Graphics Processing Units. We will mostly focus on Nvidia GPUs
# based on the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) Julia
# package.
#
# !!! info "GPU parallelism not supported everywhere"
#     GPU support is still a relatively new feature in DFTK.
#     While basic SCF computations and e.g. forces are supported,
#     this is not yet the case for all parts of the code.
#     In most cases there is no intrinsic limitation and typically it only takes
#     minor code modification to make it work on GPUs.
#     If you require GPU support in one of our routines, where this is not
#     yet supported, feel free to open an issue on github or otherwise get in touch.
#
# First we load required packages.

using AtomsBuilder
using DFTK
using PseudoPotentialData

# Next we set up a PBE model for bulk silicon.
# Note that this proceeds exactly as in the CPU case
# (see also our [Tutorial](@ref)).

silicon = bulk(:Si)

model  = model_DFT(silicon;
                   functionals=PBE(),
                   pseudopotentials=PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"))

# Now comes the selection of the computational architecture.
# This effectively makes the choice, whether the computation will be run
# on the CPU or on a GPU. For example

using CUDA
architecture = DFTK.GPU(CuArray)

# selects the computation to run on an Nvidia GPU while

using AMDGPU
architecture = DFTK.GPU(ROCArray)

# would select to run on an AMD GPU.
#
# To make sure this script runs on the github CI (where we don't have GPUs
# available) we check for the availability of GPUs before selecting an
# architecture:

architecture = has_cuda() ? DFTK.GPU(CuArray) : DFTK.CPU()

# Based on the `architecture` we construct a `PlaneWaveBasis` object:

basis  = PlaneWaveBasis(model; Ecut=30, kgrid=(5, 5, 5), architecture)
nothing  # hide

# From here again the calculation proceeds identical to the plain
# CPU case by running the SCF and optionally force calculation.

scfres = self_consistent_field(basis; tol=1e-6)
compute_forces(scfres)

# Some words of warning:
#
# !!! warning "GPU performance"
#     Our current (February 2025) benchmarks show DFTK to have reasonable performance
#     on Nvidia / CUDA GPUs with a 50-fold to 100-fold speed-up over single-threaded
#     CPU execution. However, support on AMD GPUs has been less benchmarked and
#     there are likely rough edges. Overall this feature is relatively new
#     and we appreciate any experience reports or bug reports.
