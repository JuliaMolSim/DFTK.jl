# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual
using DFTK

function make_basis(a)
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    terms = [
        Kinetic(),
        AtomicLocal(),
        AtomicNonlocal(),
        Ewald(),
        PspCorrection()
    ]
    model = Model(lattice; atoms=atoms, terms=terms, symmetries=false)
    kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
    Ecut = 15          # kinetic energy cutoff in Hartree
    PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=[32, 32, 32])
end

a = 10.26

# scfres = self_consistent_field(basis, tol=1e-8) # LoadError: Unable to find non-fractional occupations that have the correct number of electrons. You should add a temperature.
# try a bogus tolerance for debugging
scfres = self_consistent_field(make_basis(a), tol=1e-4)

function compute_energy(scfres_ref, a)
    basis = make_basis(a)
    energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=scfres_ref.ρ)
    energies.total
end

compute_energy(a) = compute_energy(scfres, a)
compute_energy(10.26)

import FiniteDiff
FiniteDiff.finite_difference_derivative(compute_energy, 10.26) # -2.948556665633414e9 

using ForwardDiff
ForwardDiff.derivative(compute_energy, 10.26) # NaN

# using BenchmarkTools
# @btime compute_energy(10.26)                                           # 19.513 ms ( 60004 allocations:  8.15 MiB)
# @btime FiniteDiff.finite_difference_derivative(compute_energy, 10.26)  # 39.317 ms (120012 allocations: 16.29 MiB)
# @btime ForwardDiff.derivative(compute_energy, 10.26)                   # 80.757 ms (543588 allocations: 31.91 MiB)

#===#
# debug NaN in AtomicNonlocal ForwardDiff

# Bits for x86 FPU control word
const FE_INVALID    = 0x1
const FE_DIVBYZERO  = 0x4
const FE_OVERFLOW   = 0x8
const FE_UNDERFLOW  = 0x10
const FE_INEXACT    = 0x20

fpexceptions() = ccall(:fegetexcept, Cint, ())

function setfpexceptions(f, mode)
    prev = ccall(:feenableexcept, Cint, (Cint,), mode)
    try
        f()
    finally
        ccall(:fedisableexcept, Cint, (Cint,), mode & ~prev)
    end
end

setfpexceptions(FE_DIVBYZERO) do
    FiniteDiff.finite_difference_derivative(compute_energy, 10.26)
end

setfpexceptions(FE_DIVBYZERO) do
    ForwardDiff.derivative(compute_energy, 10.26)  
end
# ERROR: LoadError: DivideError: integer division error
# Stacktrace:
#   [1] /
#     @ ./math.jl:0 [inlined]
#   [2] inv
#     @ ./number.jl:217 [inlined]
#   [3] sqrt
#     @ ~/.julia/packages/ForwardDiff/QOqCN/src/dual.jl:203 [inlined]
#   [4] macro expansion
#     @ ~/.julia/packages/StaticArrays/NTbHj/src/linalg.jl:225 [inlined]
#   [5] _norm
#     @ ~/.julia/packages/StaticArrays/NTbHj/src/linalg.jl:213 [inlined]
#   [6] norm
#     @ ~/.julia/packages/StaticArrays/NTbHj/src/linalg.jl:212 [inlined]
#   [7] (::AtomicLocal)(basis::PlaneWaveBasis{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/terms/local.jl:85
#   [8] macro expansion
#     @ ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:190 [inlined]
#   [9] macro expansion
#     @ ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:253 [inlined]
#  [10] (::DFTK.var"#77#79"{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Int64})()
#     @ DFTK ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:237
#  [11] timeit(f::DFTK.var"#77#79"{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Int64}, to::TimerOutputs.TimerOutput, label::String)
#     @ TimerOutputs ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:285
#  [12] #PlaneWaveBasis#76
#     @ ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:236 [inlined]
#  [13] PlaneWaveBasis(model::Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Ecut::Int64; kgrid::Vector{Int64}, kshift::Vector{Int64}, use_symmetry::Bool, kwargs::Base.Iterators.Pairs{Symbol, Vector{Int64}, Tuple{Symbol}, NamedTuple{(:fft_size,), Tuple{Vector{Int64}}}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:293
#  [14] make_basis(a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:21
#  [15] compute_energy(scfres_ref::NamedTuple{(:ham, :basis, :energies, :converged, :ρ, :eigenvalues, :occupation, :εF, :n_iter, :n_ep_extra, :ψ, :diagonalization, :stage), Tuple{Hamiltonian, PlaneWaveBasis{Float64}, Energies{Float64}, Bool, Array{Float64, 4}, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Float64, Int64, Int64, Vector{Matrix{ComplexF64}}, NamedTuple{(:λ, :X, :residual_norms, :iterations, :converged, :n_matvec), Tuple{Vector{Vector{Float64}}, Vector{Matrix{ComplexF64}}, Vector{Vector{Float64}}, Vector{Int64}, Bool, Int64}}, Symbol}}, a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:31
#  [16] compute_energy(a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:36
#  [17] derivative
#     @ ~/.julia/packages/ForwardDiff/QOqCN/src/derivative.jl:14 [inlined]
#  [18] (::var"#15#16")()
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:76
#  [19] setfpexceptions(f::var"#15#16", mode::UInt8)
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:65
#  [20] top-level scope
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:75
