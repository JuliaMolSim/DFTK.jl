# # Using DFTK on GPUs
#
# In this example we will look at how DFTK can be used on
# Graphics Processing Units.
# In its current state, runs based on Nvidia GPUs
# using the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) Julia
# package are better supported. Running on AMD GPUs is also possible 
# with the [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) package,
# albeit with lower performance.
#
# !!! info "GPU parallelism not supported everywhere"
#     Not all features of DFTK are ported to the GPU. SCF and forces
#     with standard Libxc functionals are fully supported. Stresses
#     and response calculations are work in progress as of December 2025.
#     In most cases there is no intrinsic limitation and typically it only takes
#     minor code modifications to make it work on GPUs (and some extra work for 
#     optimization). If you require GPU support in one of our routines, where this is not
#     yet supported, feel free to open an issue on github or otherwise get in touch.
#

using AtomsBuilder
using DFTK
using PseudoPotentialData

# **Model setup.** First step is to set up a [`Model`](@ref) in DFTK.
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
# If you install the CUDA package, all required Nvidia cuda libraries
# will be automatically downloaded. So literally, the only thing
# you have to do is:
using CUDA
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
#     Our current (December 2025) benchmarks show DFTK to have reasonable performance
#     on Nvidia / CUDA GPUs with up to a 100-fold speed-up over single-threaded
#     CPU execution (SCF + forces). A lot of work has been done to stabilize
#     the AMDGPU implementation as well, but performance is typically lower 
#     (~20x speedup). There may still be rough edges, and we would appreciate
#     experience or bug reports.
