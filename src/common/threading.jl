import FFTW
using LinearAlgebra

function setup_threading(;n_fft=1, n_blas=nothing)
    n_julia = Threads.nthreads()

    # Note: Threading in MKL and openblas behave rather differently.
    # See the details in
    # https://carstenbauer.github.io/ThreadPinning.jl/dev/explanations/blas/
    if isnothing(n_blas)
        libblas = basename(first(BLAS.get_config().loaded_libs).libname)
        if contains(libblas, "openblas")
            n_blas = 1
        else
            n_blas = n_julia
        end
    end
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
