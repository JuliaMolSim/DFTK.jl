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
    # Custom reduction operators are not natively supported on aarch64 (see
    # https://github.com/JuliaParallel/MPI.jl/issues/404). However, since
    # MPI.jl v0.20.22, there is an official workaround. Custom operators
    # can be statically registered:

    MPI.@RegisterOp(min, Bool)
    MPI.@RegisterOp(max, Bool)

    type_symbol = eval(Vec3{T}) where {T}
    MPI.@RegisterOp(+, type_symbol)

    type_symbol = eval(ForwardDiff.Dual{T, U, V}) where {T, U, V}
    MPI.@RegisterOp(+, type_symbol)
    MPI.@RegisterOp(min, type_symbol)
    MPI.@RegisterOp(max, type_symbol)
end

mpi_dot(x1, x2, comm::MPI.Comm) = mpi_sum(dot(x1, x2), comm)
mpi_norm(x, comm::MPI.Comm) = sqrt(mpi_sum(norm2(x), comm))
