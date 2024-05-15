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
Parallelize a loop, calling `fun(i)` for side effects for all i in `range`.
If allocate_local_storage is not nothing, `fun` is called as `fun(i, st)` where
`st` is a thread-local temporary storage allocated by `allocate_local_storage()`.
"""
function parallel_loop_over_range(fun, range; allocate_local_storage=nothing)
    nthreads = @load_preference("DFTK_threads", Threads.nthreads())
    if !isnothing(allocate_local_storage)
        storages = [allocate_local_storage() for _ = 1:nthreads]
    else
        storages = nothing
    end
    parallel_loop_over_range(fun, range, storages)
end
# private interface to be called
function parallel_loop_over_range(fun, range, storages)
    nthreads = length(storages)
    chunk_length = cld(length(range), nthreads)

    # this tensorized if is ugly, but this is potentially
    # performance critical and factoring it is more trouble
    # than it's worth
    if nthreads == 1
        for i in range
            if isnothing(storages)
                fun(i)
            else
                fun(i, storages[1])
            else
        end
    elseif length(range) == 0
        # do nothing
    else
        @sync for (ichunk, chunk) in enumerate(Iterators.partition(range, chunk_length))
        Threads.@spawn for i in chunk  # spawn a task per chunk
            if isnothing(storages)
                fun(i)
            else
                fun(i, storages[ichunk])
            else
        end
    end

    return storages
end
