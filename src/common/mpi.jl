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

mpi_sum( arr, comm::MPI.Comm) = MPI.Allreduce( arr, +, comm)
mpi_sum!(arr, comm::MPI.Comm) = MPI.Allreduce!(arr, +, comm)
mpi_min( arr, comm::MPI.Comm) = MPI.Allreduce( arr, min, comm)
mpi_min!(arr, comm::MPI.Comm) = MPI.Allreduce!(arr, min, comm)
mpi_max( arr, comm::MPI.Comm) = MPI.Allreduce( arr, max, comm)
mpi_max!(arr, comm::MPI.Comm) = MPI.Allreduce!(arr, max, comm)
mpi_average(arr, comm::MPI.Comm) = mpi_sum(arr, comm) ./ mpi_nprocs(comm)
mpi_average!(arr, comm::MPI.Comm) = (mpi_sum!(arr, comm); arr ./= mpi_nprocs(comm))

"""
Splits N work units between the processes and returns the slice of 1:N handled by the current process
"""
function mpi_split_work(comm, N)
    nprocs = mpi_nprocs()
    @assert nprocs <= N

    my_rank = MPI.Comm_rank(comm)  # 0-based
    N_per_proc = div(N, nprocs, RoundUp)
    ibeg = my_rank * N_per_proc + 1
    iend = (my_rank+1) * N_per_proc
    if iend > N
        iend = N  # last process is slacking off.
        # This will result in work distribution like (eg) 3 3 3 1: it's not bad but 3 3 2 2 might be a bit better
        # TODO optimize this better. Eg see
        # https://stackoverflow.com/questions/15658145/how-to-share-work-roughly-evenly-between-processes-in-mpi-despite-the-array-size
    end
    # println("Process $(my_rank+1)/$(nprocs) computing $(ibeg:iend)")
    ibeg:iend
end
