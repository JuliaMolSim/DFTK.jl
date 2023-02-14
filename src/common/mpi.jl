# Convenience functions for working with MPI
import MPI

"""
Number of processors used in MPI. Can be called without ensuring initialization.
"""
mpi_nprocs(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_size(comm))
mpi_master(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_rank(comm) == 0)

@static if Base.Sys.ARCH == :aarch64
    # Custom reduction operators are not supported on aarch64 (see
    # https://github.com/JuliaParallel/MPI.jl/issues/404), so we define fallback no-op
    # mpi_* functions to get things working while waiting for an upstream solution.
    for fun in (:mpi_sum, :mpi_sum!, :mpi_min, :mpi_min!, :mpi_max, :mpi_max!,
                :mpi_mean, :mpi_mean!)
        @eval $fun(arr, ::MPI.Comm) = arr
    end
else
    mpi_sum(  arr, comm::MPI.Comm) = MPI.Allreduce( arr,   +, comm)
    mpi_sum!( arr, comm::MPI.Comm) = MPI.Allreduce!(arr,   +, comm)
    mpi_min(  arr, comm::MPI.Comm) = MPI.Allreduce( arr, min, comm)
    mpi_min!( arr, comm::MPI.Comm) = MPI.Allreduce!(arr, min, comm)
    mpi_max(  arr, comm::MPI.Comm) = MPI.Allreduce( arr, max, comm)
    mpi_max!( arr, comm::MPI.Comm) = MPI.Allreduce!(arr, max, comm)
    mpi_mean( arr, comm::MPI.Comm) = mpi_sum(arr, comm) ./ mpi_nprocs(comm)
    mpi_mean!(arr, comm::MPI.Comm) = (mpi_sum!(arr, comm); arr ./= mpi_nprocs(comm))
end