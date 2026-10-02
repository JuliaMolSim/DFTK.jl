# # Using DFTK on GPUs
#
# In this example we will look how DFTK can be used on
# Graphics Processing Units.
# In its current state runs based on Nvidia GPUs
# using the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) Julia
# package are better supported and there are considerably less rough
# edges.
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

using AtomsBuilder
using DFTK
using PseudoPotentialData

# **Model setup.** First step is to setup a [`Model`](@ref) in DFTK.
# This proceeds exactly as in the standard CPU case
# (see also our [Tutorial](@ref)).

silicon = bulk(:Si)

model  = model_DFT(silicon;
                   functionals=PBE(),
                   pseudopotentials=PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"))
nothing  # hide

# Next is the selection of the computational architecture.
# This effectively makes the choice, whether the computation will be run
# on the CPU or on a GPU.
#
# **Nvidia GPUs.**
# Supported via [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).
# Right now `Libxc` only supports CUDA 11,
# so we need to explicitly request the 11.8 CUDA runtime:
using CUDA
CUDA.set_runtime_version!(v"11.8")  # Note: This requires a restart of Julia
architecture = DFTK.GPU(CuArray)

# **AMD GPUs.** Supported via [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl).
# Here you need to [install ROCm](https://rocm.docs.amd.com/) manually.
# With that in place you can then select:

using AMDGPU
architecture = DFTK.GPU(ROCArray)

# **Portable architecture selection.**
# To make sure this script runs on the github CI (where we don't have GPUs
# available) we check for the availability of GPUs before selecting an
# architecture:

architecture = has_cuda() ? DFTK.GPU(CuArray) : DFTK.CPU()

# **Basis and SCF.**
# Based on the `architecture` we construct a [`PlaneWaveBasis`](@ref) object
# as usual:

basis  = PlaneWaveBasis(model; Ecut=30, kgrid=(5, 5, 5), architecture)
nothing  # hide

# ... and run the SCF and some post-processing:

scfres = self_consistent_field(basis; tol=1e-6)
compute_forces(scfres)

# !!! warning "GPU performance"
#     Our current (February 2025) benchmarks show DFTK to have reasonable performance
#     on Nvidia / CUDA GPUs with a 50-fold to 100-fold speed-up over single-threaded
#     CPU execution. However, support on AMD GPUs has been less benchmarked and
#     there are likely rough edges. Overall this feature is relatively new
#     and we appreciate any experience reports or bug reports.
