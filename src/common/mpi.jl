# Convenience functions for working with MPI
import MPI

"""
Number of processors used in MPI. Can be called without ensuring initialization.
"""
mpi_nprocs(comm::MPI.Comm) = (MPI.Init(); MPI.Comm_size(comm))
mpi_master(comm::MPI.Comm) = (MPI.Init(); MPI.Comm_rank(comm) == 0)

# Calling MPI without explicit communcator is deprecated
@deprecate mpi_nprocs() mpi_nprocs(MPI.COMM_WORLD)
@deprecate mpi_master() mpi_master(MPI.COMM_WORLD)

# GPU-aware MPI requires device synchronization to avoid race conditions. No overhead on CPU.
sync(arr::AbstractGPUArray) = synchronize_device(architecture(arr))
sync(arr) = nothing

# Wrappers around standard MPI operations. Avoid using MPI.COMM_WORLD unless strictly necessary.
mpi_sum(  arr, comm::MPI.Comm) = (sync(arr); MPI.Allreduce( arr,   +, comm))
mpi_sum!( arr, comm::MPI.Comm) = (sync(arr); MPI.Allreduce!(arr,   +, comm))
mpi_min(  arr, comm::MPI.Comm) = (sync(arr); MPI.Allreduce( arr, min, comm))
mpi_min!( arr, comm::MPI.Comm) = (sync(arr); MPI.Allreduce!(arr, min, comm))
mpi_max(  arr, comm::MPI.Comm) = (sync(arr); MPI.Allreduce( arr, max, comm))
mpi_max!( arr, comm::MPI.Comm) = (sync(arr); MPI.Allreduce!(arr, max, comm))
mpi_mean( arr, comm::MPI.Comm) = mpi_sum(arr, comm) ./ mpi_nprocs(comm)
mpi_mean!(arr, comm::MPI.Comm) = (mpi_sum!(arr, comm); arr ./= mpi_nprocs(comm))

mpi_bcast(arr, comm::MPI.Comm; root::Int=0) = (sync(arr); MPI.bcast(arr, root, comm))
mpi_bcast(arr, root::Int, comm::MPI.Comm) = (sync(arr); MPI.bcast(arr, root, comm))
mpi_bcast!(arr, comm::MPI.Comm; root::Int=0) = (sync(arr); MPI.Bcast!(arr, root, comm))
mpi_bcast!(arr, root::Int, comm::MPI.Comm) = (sync(arr); MPI.Bcast!(arr, root, comm))
mpi_barrier(comm::MPI.Comm) = MPI.Barrier(comm)

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
