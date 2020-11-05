# Convenience functions for working with MPI
using MPI

"""
Initialize MPI. Must be called before doing any non-trivial MPI work
(even in the single-process case). Unlike the MPI.Init() function,
this can be called multiple times.
"""
function mpi_ensure_initialized()
    # MPI Thread level 3 means that the environment is multithreaded, but that only
    # one thread will call MPI at once
    # see https://www.open-mpi.org/doc/current/man3/MPI_Init_thread.3.php#toc7
    # TODO look more closely at interaction between MPI and threads
    MPI.Initialized() || MPI.Init_thread(MPI.ThreadLevel(3))
end

"""
Number of processors used in MPI. Can be called without ensuring initialization.
"""
mpi_nprocs(comm=MPI.COMM_WORLD)  = (mpi_ensure_initialized(); MPI.Comm_size(comm))
mpi_master() = (mpi_ensure_initialized(); MPI.Comm_rank(MPI.COMM_WORLD) == 0)

mpi_sum( arr, comm::MPI.Comm)  = MPI.Allreduce( arr, +, comm)
mpi_sum!(arr, comm::MPI.Comm)  = MPI.Allreduce!(arr, +, comm)
mpi_min( arr, comm::MPI.Comm)  = MPI.Allreduce( arr, min, comm)
mpi_min!(arr, comm::MPI.Comm)  = MPI.Allreduce!(arr, min, comm)
mpi_max( arr, comm::MPI.Comm)  = MPI.Allreduce( arr, max, comm)
mpi_max!(arr, comm::MPI.Comm)  = MPI.Allreduce!(arr, max, comm)
mpi_mean(arr, comm::MPI.Comm)  = mpi_sum(arr, comm) ./ mpi_nprocs(comm)
mpi_mean!(arr, comm::MPI.Comm) = (mpi_sum!(arr, comm); arr ./= mpi_nprocs(comm))

"""
Splits an iterator evenly between the processes of `comm` and returns the part handled
by the current process.
"""
function mpi_split_iterator(itr, comm)
    nprocs = mpi_nprocs(comm)
    @assert nprocs <= length(itr)
    split_evenly(itr, nprocs)[1 + MPI.Comm_rank(comm)]  # MPI ranks are 0-based
end
