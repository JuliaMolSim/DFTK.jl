# Convenience functions for working with MPI
using MPI

"""
Initialize MPI. Must be called before doing any non-trivial MPI work
(even in the single-process case). Unlike the MPI.Init() function,
this can be called multiple times.
"""
function mpi_ensure_initialized()
    # TODO look more closely at interaction between MPI and threads
    MPI.Initialized() || MPI.Init_thread(MPI.ThreadLevel(3))
end

"""
Number of processors used in MPI. Can be called without ensuring initialization.
"""
mpi_nprocs() = mpi_nprocs(MPI.COMM_WORLD)
mpi_nprocs(comm) = (mpi_ensure_initialized(); MPI.Comm_size(comm))
mpi_master() = (mpi_ensure_initialized(); MPI.Comm_rank(MPI.COMM_WORLD) == 0)

mpi_sum(comm::MPI.Comm, arr) = MPI.Allreduce(arr, +, comm)
mpi_sum!(comm::MPI.Comm, arr) = MPI.Allreduce!(arr, +, comm)
mpi_min(comm::MPI.Comm, arr) = MPI.Allreduce(arr, min, comm)
mpi_min!(comm::MPI.Comm, arr) = MPI.Allreduce!(arr, min, comm)
mpi_max(comm::MPI.Comm, arr) = MPI.Allreduce(arr, max, comm)
mpi_max!(comm::MPI.Comm, arr) = MPI.Allreduce!(arr, max, comm)
mpi_average(comm::MPI.Comm, arr) = mpi_sum(comm, arr) ./ mpi_nprocs(comm)
mpi_average!(comm::MPI.Comm, arr) = (mpi_sum!(comm, arr); arr ./= mpi_nprocs(comm))
