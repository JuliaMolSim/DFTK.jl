import FFTW
using LinearAlgebra

function setup_threading(;n_fft=1, n_blas=Threads.nthreads())
    n_julia = Threads.nthreads()
    FFTW.set_num_threads(n_fft)
    BLAS.set_num_threads(n_blas)
    mpi_master() && @info "Threading setup" n_fft n_blas n_julia
    nothing
end
