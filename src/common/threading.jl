import FFTW
using LinearAlgebra

function setup_threading(; n_fft=1, n_blas=Threads.nthreads())
    n_julia = Threads.nthreads()
    FFTW.set_num_threads(n_fft)
    BLAS.set_num_threads(n_blas)
    mpi_master() && @info "Threading setup:" n_fft n_blas n_julia
end

"""
Convenience function to disable all threading in DFTK and assert that Julia threading
is off as well.
"""
function disable_threading()
    n_julia = Threads.nthreads()
    n_julia > 1 && mpi_master() && error(
        "Julia currently uses $n_julia threads. Ensure that the environment variable " *
        "JULIA_NUM_THREADS is unset and julia does not get the `-t` flag passed."
    )
    @assert n_julia == 1  # To exit in non-master MPI nodes
    setup_threading(;n_fft=1, n_blas=1)
end

# Parallelization loop breaking range into chunks.
function parallel_loop_over_range(fun, storages::AbstractVector, range)
    chunk_length = cld(length(range), Threads.nthreads())

    iszero(chunk_length) && return storages

    @sync for (ichunk, chunk) in enumerate(Iterators.partition(range, chunk_length))
        Threads.@spawn for idc in chunk  # spawn a task per chunk
            fun(storages[ichunk], idc)
        end
    end

    return storages
end
function parallel_loop_over_range(fun, allocate_local_storage::Function, range)
    storages = [allocate_local_storage() for _ = 1:Threads.nthreads()]
    parallel_loop_over_range(fun, storages, range)
end
