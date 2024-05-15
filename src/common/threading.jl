import FFTW
using LinearAlgebra

function setup_threading(; n_fft=1, n_blas=Threads.nthreads(), n_DFTK=nothing)
    if n_DFTK != nothing
        set_DFTK_threads!(n_DFTK)
    end
    n_DFTK = @load_preference("DFTK_threads", Threads.nthreads())
    FFTW.set_num_threads(n_fft)
    BLAS.set_num_threads(n_blas)
    mpi_master() && @info "Threading setup: $n_julia Julia threads, $n_DFTK DFTK threads, $n_fft FFT threads, $n_blas BLAS threads"

end

"""
Convenience function to disable all threading in DFTK and assert that Julia threading
is off as well.
"""
function disable_threading()
    setup_threading(;n_fft=1, n_blas=1, n_DFTK=1)
end

function set_DFTK_threads!(n)
    if @load_preference("DFTK_threads", nothing) != n
        @info "DFTK_threads preference changed. Restart julia to see the effect."
    end
    @set_preferences!("DFTK_threads" => n)
end
function set_DFTK_threads!()
    @delete_preferences!("DFTK_threads")
end

"""
Parallelize a loop, calling `fun(st, i)` for side effects for all i in `range`. 
`st` is a thread-local temporary storage allocated by `allocate_local_storage()`.
"""
function parallel_loop_over_range(fun, allocate_local_storage::Function, range)
    nthreads = @load_preference("DFTK_threads", Threads.nthreads())
    storages = [allocate_local_storage() for _ = 1:nthreads]
    parallel_loop_over_range(fun, storages, range)
end
function parallel_loop_over_range(fun, storages::AbstractVector, range)
    nthreads = length(storages)
    chunk_length = cld(length(range), nthreads)

    if nthreads == 1
        for i in range
            fun(storages[1], i)
        end
    elseif length(range) == 0
        # do nothing
    else
        @sync for (ichunk, chunk) in enumerate(Iterators.partition(range, chunk_length))
        Threads.@spawn for idc in chunk  # spawn a task per chunk
            fun(storages[ichunk], idc)
        end
    end

    return storages
end
