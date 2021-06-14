using DFTK
using Test

function make_basis(a)
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    terms = [
        Kinetic(),
        # AtomicLocal(),
        # AtomicNonlocal(),
        # Ewald(),
        # PspCorrection()
    ]
    model = Model(lattice; atoms=atoms, terms=terms, symmetries=false)
    kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
    Ecut = 15          # kinetic energy cutoff in Hartree
    PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=[32, 32, 32])
end

a = 10.26
basis = make_basis(a)

# scfres = self_consistent_field(basis, tol=1e-8) # LoadError: Unable to find non-fractional occupations that have the correct number of electrons. You should add a temperature.
# try a bogus tolerance for debugging
scfres = self_consistent_field(basis, tol=1e9)

function compute_energy(scfres_ref, a)
    basis = make_basis(a)
    energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=scfres_ref.ρ)
    energies.total
end

compute_energy(scfres, 10.26)

import FiniteDiff
FiniteDiff.finite_difference_derivative(a -> compute_energy(scfres, a), 10.26) # -11.113131188820518 

###
### Forward mode
###

using ForwardDiff
ForwardDiff.derivative(a -> compute_energy(scfres, a), 10.26)
# ERROR: LoadError: MethodError: no method matching build_fft_plans(::Type{ForwardDiff.Dual{ForwardDiff.Tag{var"#23#24", Float64}, Float64, 1}}, ::Tuple{Int64, Int64, Int64})
# Closest candidates are:
#   build_fft_plans(::Union{Type{Float32}, Type{Float64}}, ::Any) at /home/niku/.julia/dev/DFTK.jl/src/fft.jl:154
# Stacktrace:
#   [1] macro expansion
#     @ ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:196 [inlined]
#   [2] (::DFTK.var"#62#64"{ForwardDiff.Dual{ForwardDiff.Tag{var"#23#24", Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{var"#23#24", Float64}, Float64, 1}}, Int64})()
#     @ DFTK ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:237
#   [3] timeit(f::DFTK.var"#62#64"{ForwardDiff.Dual{ForwardDiff.Tag{var"#23#24", Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{var"#23#24", Float64}, Float64, 1}}, Int64}, to::TimerOutputs.TimerOutput, label::String)
#     @ TimerOutputs ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:285
#   [4] PlaneWaveBasis(model::Model{ForwardDiff.Dual{ForwardDiff.Tag{var"#23#24", Float64}, Float64, 1}}, Ecut::Int64, kcoords::Vector{StaticArrays.SVector{3, Rational{Int64}}}, ksymops::Vector{Vector{Tuple{StaticArrays.SMatrix{3, 3, Int64, 9}, StaticArrays.SVector{3, Float64}}}}, symmetries::Vector{Tuple{StaticArrays.SMatrix{3, 3, Int64, 9}, StaticArrays.SVector{3, Float64}}}; fft_size::Vector{Int64}, variational::Bool, optimize_fft_size::Bool, supersampling::Int64, kgrid::Vector{Int64}, kshift::Vector{Int64}, comm_kpts::MPI.Comm)
#     @ DFTK ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:236
#   [5] PlaneWaveBasis(model::Model{ForwardDiff.Dual{ForwardDiff.Tag{var"#23#24", Float64}, Float64, 1}}, Ecut::Int64; kgrid::Vector{Int64}, kshift::Vector{Int64}, use_symmetry::Bool, kwargs::Base.Iterators.Pairs{Symbol, Vector{Int64}, Tuple{Symbol}, NamedTuple{(:fft_size,), Tuple{Vector{Int64}}}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:286
#   [6] make_basis(a::ForwardDiff.Dual{ForwardDiff.Tag{var"#23#24", Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:20
#   [7] compute_energy(scfres_ref::NamedTuple{(:ham, :basis, :energies, :converged, :ρ, :eigenvalues, :occupation, :εF, :n_iter, :n_ep_extra, :ψ, :diagonalization, :stage), Tuple{Hamiltonian, PlaneWaveBasis{Float64}, Energies{Float64}, Bool, Array{Float64, 4}, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Float64, Int64, Int64, Vector{Matrix{ComplexF64}}, NamedTuple{(:λ, :X, :residual_norms, :iterations, :converged, :n_matvec), Tuple{Vector{Vector{Float64}}, Vector{Matrix{ComplexF64}}, Vector{Vector{Float64}}, Vector{Int64}, Bool, Int64}}, Symbol}}, a::ForwardDiff.Dual{ForwardDiff.Tag{var"#23#24", Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:31
#   [8] (::var"#23#24")(a::ForwardDiff.Dual{ForwardDiff.Tag{var"#23#24", Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:46
#   [9] derivative(f::var"#23#24", x::Float64)
#     @ ForwardDiff ~/.julia/packages/ForwardDiff/m7cm5/src/derivative.jl:14
#  [10] top-level scope
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:46
