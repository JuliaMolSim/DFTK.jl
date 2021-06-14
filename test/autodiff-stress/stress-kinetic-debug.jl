using DFTK
using Test

a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_atomic_debug(lattice, atoms, symmetries=false)
kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 15          # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=[32, 32, 32])

# scfres = self_consistent_field(basis, tol=1e-8) # LoadError: Unable to find non-fractional occupations that have the correct number of electrons. You should add a temperature.

# try a bogus tolerance for debugging
scfres = self_consistent_field(basis, tol=1e9)

function compute_energy(scfres_ref, a)
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]

    model = model_atomic_debug(lattice, atoms, symmetries=false)
    kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
    Ecut = 15           # kinetic energy cutoff in Hartree
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=[32, 32, 32])

    energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=scfres_ref.ρ)
    energies.total
end

compute_energy(scfres, 10.26)

function compute_stress(scfres_ref, a)
    Inf # TODO implement
end
@test compute_stress(scfres, a) ≈ FiniteDiff.finite_difference_derivative(a -> compute_energy(scfres, a), a) # -1.411


###
### Forward mode
###

using ForwardDiff
ForwardDiff.derivative(a -> compute_energy(scfres, a), 10.26)
# ERROR: LoadError: MethodError: no method matching next_working_fft_size(::Type{ForwardDiff.Dual{ForwardDiff.Tag{var"#7#8", Float64}, Float64, 1}}, ::Int64)
# Closest candidates are:
#   next_working_fft_size(::Type{Float32}, ::Any) at /home/niku/.julia/dev/DFTK.jl/src/fft.jl:167
#   next_working_fft_size(::Type{Float64}, ::Any) at /home/niku/.julia/dev/DFTK.jl/src/fft.jl:168
# Stacktrace:
#   [1] _broadcast_getindex_evalf
#     @ ./broadcast.jl:648 [inlined]
#   [2] _broadcast_getindex
#     @ ./broadcast.jl:631 [inlined]
#   [3] getindex
#     @ ./broadcast.jl:575 [inlined]
#   [4] copy
#     @ ./broadcast.jl:922 [inlined]
#   [5] materialize
#     @ ./broadcast.jl:883 [inlined]
#   [6] validate_or_compute_fft_size(model::Model{ForwardDiff.Dual{ForwardDiff.Tag{var"#7#8", Float64}, Float64, 1}}, fft_size::Vector{Int64}, Ecut::Int64, supersampling::Int64, variational::Bool, optimize_fft_size::Bool, kcoords::Vector{StaticArrays.SVector{3, Rational{Int64}}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/fft.jl:139
#   [7] macro expansion
#     @ ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:193 [inlined]
#   [8] (::DFTK.var"#62#64"{ForwardDiff.Dual{ForwardDiff.Tag{var"#7#8", Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{var"#7#8", Float64}, Float64, 1}}, Int64})()
#     @ DFTK ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:237
#   [9] timeit(f::DFTK.var"#62#64"{ForwardDiff.Dual{ForwardDiff.Tag{var"#7#8", Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{var"#7#8", Float64}, Float64, 1}}, Int64}, to::TimerOutputs.TimerOutput, label::String)
#     @ TimerOutputs ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:285
#  [10] PlaneWaveBasis(model::Model{ForwardDiff.        Dual{ForwardDiff.Tag{var"#7#8", Float64}, Floa        t64, 1}}, Ecut::Int64, kcoords::Vector{StaticA        rrays.SVector{3, Rational{Int64}}}, ksymops::Vector{Vector{Tuple{StaticArrays.SMatrix{3, 3, Int64, 9}, StaticArrays.SVector{3, Float64}}}}, symmetries::Vector{Tuple{StaticArrays.SMatrix{3, 3, Int64, 9}, StaticArrays.SVector{3, Float64}}}; fft_size::Vector{Int64}, variational::Bool, optimize_fft_size::Bool, supersampling::Int64, kgrid::Vector{Int64}, kshift::Vector{Int64}, comm_kpts::MPI.Comm)
#     @ DFTK ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:236
#  [11] PlaneWaveBasis(model::Model{ForwardDiff.Dual{ForwardDiff.Tag{var"#7#8", Float64}, Float64, 1}}, Ecut::Int64; kgrid::Vector{Int64}, kshift::Vector{Int64}, use_symmetry::Bool, kwargs::Base.Iterators.Pairs{Symbol, Vector{Int64}, Tuple{Symbol}, NamedTuple{(:fft_size,), Tuple{Vector{Int64}}}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:286
#  [12] compute_energy(scfres_ref::NamedTuple{(:ham, :basis, :energies, :converged, :ρ, :eigenvalues, :occupation, :εF, :n_iter, :n_ep_extra, :ψ, :diagonalization, :stage), Tuple{Hamiltonian, PlaneWaveBasis{Float64}, Energies{Float64}, Bool, Array{Float64, 4}, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Float64, Int64, Int64, Vector{Matrix{ComplexF64}}, NamedTuple{(:λ, :X, :residual_norms, :iterations, :converged, :n_matvec), Tuple{Vector{Vector{Float64}}, Vector{Matrix{ComplexF64}}, Vector{Vector{Float64}}, Vector{Int64}, Bool, Int64}}, Symbol}}, a::ForwardDiff.Dual{ForwardDiff.Tag{var"#7#8", Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-kinetic-debug.jl:31
#  [13] (::var"#7#8")(a::ForwardDiff.Dual{ForwardDiff.Tag{var"#7#8", Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-kinetic-debug.jl:50
#  [14] derivative(f::var"#7#8", x::Float64)
#     @ ForwardDiff ~/.julia/packages/ForwardDiff/m7cm5/src/derivative.jl:14
#  [15] top-level scope
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-kinetic-debug.jl:50
