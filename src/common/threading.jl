import FFTW
using LinearAlgebra

"""
Setup the number of threads used by DFTK's own threading (`n_DFTK`), by BLAS (`n_blas`) 
and by FFTW (`n_fft`). This is independent from the number of Julia threads (`Threads.nthreads()`).
DFTK and FFTW threading are upper bounded by `Threads.nthreads()`, but not BLAS,
which uses its own threading system.
By default, use 1 FFT thread, and `Threads.nthreads()` BLAS and DFTK threads.
"""
function setup_threading(; n_fft=1, n_blas=Threads.nthreads(), n_DFTK=Threads.nthreads())
    set_DFTK_threads!(n_DFTK)
    FFTW.set_num_threads(n_fft)
    BLAS.set_num_threads(n_blas)
    mpi_master() && @info "Threading setup: " Threads.nthreads() n_DFTK n_fft n_blas
end

"""
Convenience function to disable all threading in DFTK.
"""
function disable_threading()
    setup_threading(;n_fft=1, n_blas=1, n_DFTK=1)
end

# TODO: is a single write to an Int64 atomic?
const DFTK_threads = Ref(Threads.nthreads())

function set_DFTK_threads!(n)
    if n > Threads.nthreads()
        error("You set your preference for $n DFTK threads using set_DFTK_threads!, " *
              "but only ran julia with $(Threads.nthreads()) threads.")
    end
    if n <= 0
        error("You tried to set DFTK threads to $n, at least 1 is required.")
    end
    DFTK_threads[] = n
end
function get_DFTK_threads()
    DFTK_threads[]
end

"""
Parallelize a loop, calling `fun(i)` for side effects for all i in `range`.
If allocate_local_storage is not nothing, `fun` is called as `fun(i, st)` where
`st` is a thread-local temporary storage allocated by `allocate_local_storage()`.
"""
function parallel_loop_over_range(fun, range; allocate_local_storage=nothing)
    nthreads = get_DFTK_threads()
    if !isnothing(allocate_local_storage)
        storages = [allocate_local_storage() for _ = 1:nthreads]
    else
        storages = nothing
    end
    parallel_loop_over_range(fun, range, storages)
end
# private interface to be called
function parallel_loop_over_range(fun, range, storages)
    nthreads = get_DFTK_threads()
    !isnothing(storages) && @assert length(storages) >= nthreads
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
            end
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
                end
            end
        end
    end

    return storages
end
