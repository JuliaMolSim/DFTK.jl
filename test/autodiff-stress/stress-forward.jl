# Hellmann-Feynman stress
# via ForwardDiff & custom FFTW overloads on ForwardDiff.Dual
using DFTK

function make_basis(a)
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    C = ElementPsp(:C, psp=load_psp("hgh/lda/c-q4.hgh"))
    atoms = [C => [ones(3)/8]]
    terms = [
        Kinetic(),
        AtomicLocal(),
        # AtomicNonlocal(),
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
scfres = self_consistent_field(make_basis(a), tol=1e9)
# ┌ Warning: Mismatch in number of electrons
# │   sum_ρ = 1080.0455760000316
# │   sum_occupation = 4.0
# └ @ DFTK ~/.julia/dev/DFTK.jl/src/densities.jl:32
# n     Free energy       Eₙ-Eₙ₋₁     ρout-ρin   Diag
# ---   ---------------   ---------   --------   ----
#   1   -3819171908.212         NaN   2.49e+07    21.0 
# ERROR: LoadError: Unable to find non-fractional occupations that have the correct number of electrons. You should add a temperature.
# Stacktrace:
#   [1] error(s::String)
#     @ Base ./error.jl:33
#   [2] compute_occupation(basis::PlaneWaveBasis{Float64}, energies::Vector{Vector{Float64}}; temperature::Float64, smearing::DFTK.Smearing.None)
#     @ DFTK ~/.julia/dev/DFTK.jl/src/occupation.jl:77
#   [3] compute_occupation(basis::PlaneWaveBasis{Float64}, energies::Vector{Vector{Float64}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/occupation.jl:16
#   [4] next_density(ham::Hamiltonian; n_bands::Int64, ψ::Vector{Matrix{ComplexF64}}, n_ep_extra::Int64, eigensolver::Function, occupation_function::typeof(DFTK.compute_occupation), kwargs::Base.Iterators.Pairs{Symbol, Real, Tuple{Symbol, Symbol}, NamedTuple{(:miniter, :tol), Tuple{Int64, Float64}}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/scf/self_consistent_field.jl:30
#   [5] (::DFTK.var"#fixpoint_map#520"{DataType, Int64, typeof(lobpcg_hyper), Int64, DFTK.var"#determine_diagtol#515"{Float64}, Float64, SimpleMixing, DFTK.var"#is_converged#511"{Float64}, DFTK.var"#callback#510", Bool, Bool, typeof(DFTK.compute_occupation), PlaneWaveBasis{Float64}})(ρin::Array{Float64, 4})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/scf/self_consistent_field.jl:98
#   [6] (::DFTK.var"#487#490"{DFTK.var"#fixpoint_map#520"{DataType, Int64, typeof(lobpcg_hyper), Int64, DFTK.var"#determine_diagtol#515"{Float64}, Float64, SimpleMixing, DFTK.var"#is_converged#511"{Float64}, DFTK.var"#callback#510", Bool, Bool, typeof(DFTK.compute_occupation), PlaneWaveBasis{Float64}}})(x::Array{Float64, 4})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/scf/scf_solvers.jl:18
#   [7] (::NLSolversBase.var"#ff!#1"{DFTK.var"#487#490"{DFTK.var"#fixpoint_map#520"{DataType, Int64, typeof(lobpcg_hyper), Int64, DFTK.var"#determine_diagtol#515"{Float64}, Float64, SimpleMixing, DFTK.var"#is_converged#511"{Float64}, DFTK.var"#callback#510", Bool, Bool, typeof(DFTK.compute_occupation), PlaneWaveBasis{Float64}}}})(F::Array{Float64, 4}, x::Array{Float64, 4})
#     @ NLSolversBase ~/.julia/packages/NLSolversBase/geyh3/src/objective_types/inplace_factory.jl:11
#   [8] value!!(obj::NLSolversBase.NonDifferentiable{Array{Float64, 4}, Array{Float64, 4}}, F::Array{Float64, 4}, x::Array{Float64, 4})
#     @ NLSolversBase ~/.julia/packages/NLSolversBase/geyh3/src/interface.jl:166
#   [9] value!!
#     @ ~/.julia/packages/NLSolversBase/geyh3/src/interface.jl:163 [inlined]
#  [10] anderson_(df::NLSolversBase.NonDifferentiable{Array{Float64, 4}, Array{Float64, 4}}, initial_x::Array{Float64, 4}, xtol::Float64, ftol::Float64, iterations::Int64, store_trace::Bool, show_trace::Bool, extended_trace::Bool, beta::Int64, aa_start::Int64, droptol::Float64, cache::NLsolve.AndersonCache{Array{Float64, 4}, Array{Float64, 4}, Vector{Array{Float64, 4}}, Vector{Float64}, Matrix{Float64}, Matrix{Float64}})
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/solvers/anderson.jl:73
#  [11] anderson(df::NLSolversBase.NonDifferentiable{Array{Float64, 4}, Array{Float64, 4}}, initial_x::Array{Float64, 4}, xtol::Float64, ftol::Float64, iterations::Int64, store_trace::Bool, show_trace::Bool, extended_trace::Bool, beta::Int64, aa_start::Int64, droptol::Float64, cache::NLsolve.AndersonCache{Array{Float64, 4}, Array{Float64, 4}, Vector{Array{Float64, 4}}, Vector{Float64}, Matrix{Float64}, Matrix{Float64}})
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/solvers/anderson.jl:203
#  [12] anderson(df::NLSolversBase.NonDifferentiable{Array{Float64, 4}, Array{Float64, 4}}, initial_x::Array{Float64, 4}, xtol::Float64, ftol::Float64, iterations::Int64, store_trace::Bool, show_trace::Bool, extended_trace::Bool, m::Int64, beta::Int64, aa_start::Int64, droptol::Float64)
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/solvers/anderson.jl:188
#  [13] nlsolve(df::NLSolversBase.NonDifferentiable{Array{Float64, 4}, Array{Float64, 4}}, initial_x::Array{Float64, 4}; method::Symbol, xtol::Float64, ftol::Float64, iterations::Int64, store_trace::Bool, show_trace::Bool, extended_trace::Bool, linesearch::LineSearches.Static, linsolve::NLsolve.var"#29#31", factor::Float64, autoscale::Bool, m::Int64, beta::Int64, aa_start::Int64, droptol::Float64)
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/nlsolve/nlsolve.jl:30
#  [14] nlsolve(f::Function, initial_x::Array{Float64, 4}; method::Symbol, autodiff::Symbol, inplace::Bool, kwargs::Base.Iterators.Pairs{Symbol, Real, NTuple{5, Symbol}, NamedTuple{(:m, :xtol, :ftol, :show_trace, :iterations), Tuple{Int64, Float64, Float64, Bool, Int64}}})
#     @ NLsolve ~/.julia/packages/NLsolve/gJL1I/src/nlsolve/nlsolve.jl:52
#  [15] fp_solver
#     @ ~/.julia/dev/DFTK.jl/src/scf/scf_solvers.jl:18 [inlined]
#  [16] macro expansion
#     @ ~/.julia/dev/DFTK.jl/src/scf/self_consistent_field.jl:137 [inlined]
#  [17] (::DFTK.var"#518#519"{Int64, Array{Float64, 4}, Int64, DFTK.var"#fp_solver#488"{DFTK.var"#fp_solver#486#489"{Base.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}}, Int64, Symbol}}, typeof(lobpcg_hyper), Int64, DFTK.var"#determine_diagtol#515"{Float64}, Float64, SimpleMixing, DFTK.var"#is_converged#511"{Float64}, DFTK.var"#callback#510", Bool, Bool, typeof(DFTK.compute_occupation), PlaneWaveBasis{Float64}})()
#     @ DFTK ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:237
#  [18] timeit(f::DFTK.var"#518#519"{Int64, Array{Float64, 4}, Int64, DFTK.var"#fp_solver#488"{DFTK.var"#fp_solver#486#489"{Base.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}}, Int64, Symbol}}, typeof(lobpcg_hyper), Int64, DFTK.var"#determine_diagtol#515"{Float64}, Float64, SimpleMixing, DFTK.var"#is_converged#511"{Float64}, DFTK.var"#callback#510", Bool, Bool, typeof(DFTK.compute_occupation), PlaneWaveBasis{Float64}}, to::TimerOutputs.TimerOutput, label::String)
#     @ TimerOutputs ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:285
#  [19] self_consistent_field(basis::PlaneWaveBasis{Float64}; n_bands::Int64, ρ::Array{Float64, 4}, ψ::Nothing, tol::Float64, maxiter::Int64, solver::Function, eigensolver::Function, n_ep_extra::Int64, determine_diagtol::DFTK.var"#determine_diagtol#515"{Float64}, α::Float64, mixing::SimpleMixing, is_converged::DFTK.var"#is_converged#511"{Float64}, callback::DFTK.var"#callback#510", compute_consistent_energies::Bool, enforce_symmetry::Bool, occupation_function::typeof(DFTK.compute_occupation))
#     @ DFTK ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:236
#  [20] top-level scope
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:29


# function compute_energy(scfres_ref, a)
#     basis = make_basis(a)
#     energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=scfres_ref.ρ)
#     energies.total
# end

# compute_energy(a) = compute_energy(scfres, a)
# compute_energy(10.26)

# import FiniteDiff
# FiniteDiff.finite_difference_derivative(compute_energy, 10.26) # -2.948556665633414e9 

# using ForwardDiff
# ForwardDiff.derivative(compute_energy, 10.26) # -2.948556665529993e9

# using BenchmarkTools
# @btime compute_energy(10.26)                                           # 19.513 ms ( 60004 allocations:  8.15 MiB)
# @btime FiniteDiff.finite_difference_derivative(compute_energy, 10.26)  # 39.317 ms (120012 allocations: 16.29 MiB)
# @btime ForwardDiff.derivative(compute_energy, 10.26)                   # 80.757 ms (543588 allocations: 31.91 MiB)
