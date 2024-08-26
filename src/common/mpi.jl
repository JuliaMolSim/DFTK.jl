# Convenience functions for working with MPI
import MPI

"""
Number of processors used in MPI. Can be called without ensuring initialization.
"""
mpi_nprocs(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_size(comm))
mpi_master(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_rank(comm) == 0)

mpi_sum(  arr, comm::MPI.Comm) = MPI.Allreduce( arr,   +, comm)
mpi_sum!( arr, comm::MPI.Comm) = MPI.Allreduce!(arr,   +, comm)
mpi_min(  arr, comm::MPI.Comm) = MPI.Allreduce( arr, min, comm)
mpi_min!( arr, comm::MPI.Comm) = MPI.Allreduce!(arr, min, comm)
mpi_max(  arr, comm::MPI.Comm) = MPI.Allreduce( arr, max, comm)
mpi_max!( arr, comm::MPI.Comm) = MPI.Allreduce!(arr, max, comm)
mpi_mean( arr, comm::MPI.Comm) = mpi_sum(arr, comm) ./ mpi_nprocs(comm)
mpi_mean!(arr, comm::MPI.Comm) = (mpi_sum!(arr, comm); arr ./= mpi_nprocs(comm))

@static if Base.Sys.ARCH == :aarch64
    # Custom reduction operators are not supported on aarch64 (see
    # https://github.com/JuliaParallel/MPI.jl/issues/404). We define
    # temporary workarounds in order to be able to run MPI on aarch64
    # anyways. These should be removed as soon as there is an upstream fix
    include("aarch64_mpi.jl")
end