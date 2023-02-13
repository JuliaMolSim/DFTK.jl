# Convenience functions for working with MPI
import MPI

"""
Number of processors used in MPI. Can be called without ensuring initialization.
"""
mpi_nprocs(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_size(comm))
mpi_master(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_rank(comm) == 0)

@static if Base.Sys.ARCH == :aarch64
    mpi_sum(  arr, ::MPI.Comm) = arr
    mpi_sum!( arr, ::MPI.Comm) = arr
    mpi_min(  arr, ::MPI.Comm) = arr
    mpi_min!( arr, ::MPI.Comm) = arr
    mpi_max(  arr, ::MPI.Comm) = arr
    mpi_max!( arr, ::MPI.Comm) = arr
    mpi_mean( arr, ::MPI.Comm) = arr
    mpi_mean!(arr, ::MPI.Comm) = arr
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