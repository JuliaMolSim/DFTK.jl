# Convenience functions for working with MPI
import MPI

"""
Number of processors used in MPI. Can be called without ensuring initialization.
"""
mpi_nprocs(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_size(comm))
mpi_master(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_rank(comm) == 0)

mpi_sum(  arr, comm::MPI.Comm) = mpi_sum(  arr, comm, Val(Base.Sys.ARCH))
mpi_sum!( arr, comm::MPI.Comm) = mpi_sum!( arr, comm, Val(Base.Sys.ARCH))
mpi_min(  arr, comm::MPI.Comm) = mpi_min(  arr, comm, Val(Base.Sys.ARCH))
mpi_min!( arr, comm::MPI.Comm) = mpi_min!( arr, comm, Val(Base.Sys.ARCH))
mpi_max(  arr, comm::MPI.Comm) = mpi_max(  arr, comm, Val(Base.Sys.ARCH))
mpi_max!( arr, comm::MPI.Comm) = mpi_max!( arr, comm, Val(Base.Sys.ARCH))
mpi_mean( arr, comm::MPI.Comm) = mpi_mean( arr, comm, Val(Base.Sys.ARCH))
mpi_mean!(arr, comm::MPI.Comm) = mpi_mean!(arr, comm, Val(Base.Sys.ARCH))

mpi_sum(  arr, comm::MPI.Comm, ::Val{:x86_64}) = MPI.Allreduce( arr,   +, comm)
mpi_sum!( arr, comm::MPI.Comm, ::Val{:x86_64}) = MPI.Allreduce!(arr,   +, comm)
mpi_min(  arr, comm::MPI.Comm, ::Val{:x86_64}) = MPI.Allreduce( arr, min, comm)
mpi_min!( arr, comm::MPI.Comm, ::Val{:x86_64}) = MPI.Allreduce!(arr, min, comm)
mpi_max(  arr, comm::MPI.Comm, ::Val{:x86_64}) = MPI.Allreduce( arr, max, comm)
mpi_max!( arr, comm::MPI.Comm, ::Val{:x86_64}) = MPI.Allreduce!(arr, max, comm)
mpi_mean( arr, comm::MPI.Comm, ::Val{:x86_64}) = mpi_sum(arr, comm) ./ mpi_nprocs(comm)
mpi_mean!(arr, comm::MPI.Comm, ::Val{:x86_64}) = (mpi_sum!(arr, comm); arr ./= mpi_nprocs(comm))

aarch64_check_mpi(comm::MPI.Comm) = mpi_nprocs(comm) > 1 ? error("MPI not supported on aarch64") : true
mpi_sum(  arr, comm::MPI.Comm, ::Val{:aarch64}) = aarch64_check_mpi(comm) && sum(    arr)
mpi_sum!( arr, comm::MPI.Comm, ::Val{:aarch64}) = aarch64_check_mpi(comm) && sum(    arr)
mpi_min(  arr, comm::MPI.Comm, ::Val{:aarch64}) = aarch64_check_mpi(comm) && minimum(arr)
mpi_min!( arr, comm::MPI.Comm, ::Val{:aarch64}) = aarch64_check_mpi(comm) && minimum(arr)
mpi_max(  arr, comm::MPI.Comm, ::Val{:aarch64}) = aarch64_check_mpi(comm) && maximum(arr)
mpi_max!( arr, comm::MPI.Comm, ::Val{:aarch64}) = aarch64_check_mpi(comm) && maximum(arr)
mpi_mean( arr, comm::MPI.Comm, ::Val{:aarch64}) = aarch64_check_mpi(comm) && mean(   arr)
mpi_mean!(arr, comm::MPI.Comm, ::Val{:aarch64}) = aarch64_check_mpi(comm) && mean(   arr)
