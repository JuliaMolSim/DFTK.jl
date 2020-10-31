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

mpi_sum(arr, comm) = MPI.Allreduce(arr, +, comm)
mpi_sum!(arr, comm) = MPI.Allreduce!(arr, +, comm)
mpi_min(arr, comm) = MPI.Allreduce(arr, min, comm)
mpi_min!(arr, comm) = MPI.Allreduce!(arr, min, comm)
mpi_max(arr, comm) = MPI.Allreduce(arr, max, comm)
mpi_max!(arr, comm) = MPI.Allreduce!(arr, max, comm)
