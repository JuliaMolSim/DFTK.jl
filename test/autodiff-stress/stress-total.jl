using DFTK

function make_basis(a)
    lattice = a / 2 * [[0 1 1.];
                    [1 0 1.];
                    [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    model = model_atomic(lattice, atoms, symmetries=false)
    kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
    Ecut = 15          # kinetic energy cutoff in Hartree
    PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=[32,32,32])
end

a = 10.26
scfres = self_consistent_field(make_basis(a), tol=1e-8)

function compute_energy(scfres_ref, a)
    basis = make_basis(a)
    energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=scfres_ref.ρ)
    energies.total
end

compute_energy(scfres, 10.26)

import FiniteDiff
fd_stress = FiniteDiff.finite_difference_derivative(a -> compute_energy(scfres, a), a)

###
### Forward mode
###

using ForwardDiff
ForwardDiff.derivative(a -> compute_energy(scfres, a), 10.26) # NaN

###
### Reverse mode
###

using Zygote
Zygote.gradient(a -> compute_energy(scfres, a), 10.26)
# ERROR: LoadError: MethodError: no method matching zero(::String)
# Closest candidates are:
#   zero(::Union{Type{P}, P}) where P<:Dates.Period at /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/Dates/src/periods.jl:53
#   zero(::FillArrays.Ones{T, N, Axes} where Axes) where {T, N} at /home/niku/.julia/packages/FillArrays/rPtlv/src/FillArrays.jl:537
#   zero(::T) where T<:Dates.TimeType at /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/Dates/src/types.jl:423
#   ...
# Stacktrace:
#   [1] pair_getfield
#     @ ~/.julia/packages/Zygote/pM10l/src/lib/base.jl:134 [inlined]
#   [2] #2040#back
#     @ ~/.julia/packages/ZygoteRules/OjfTt/src/adjoint.jl:59 [inlined]
#   [3] Pullback
#     @ ./pair.jl:59 [inlined]
#   [4] (::typeof(∂(getindex)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#   [5] Pullback
#     @ ./abstractdict.jl:66 [inlined]
#   [6] (::typeof(∂(iterate)))(Δ::Tuple{Float64, Nothing})
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#   [7] Pullback
#     @ ./reduce.jl:60 [inlined]
#   [8] (::typeof(∂(_foldl_impl)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#   [9] Pullback
#     @ ./reduce.jl:48 [inlined]
#  [10] (::typeof(∂(foldl_impl)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [11] Pullback
#     @ ./reduce.jl:44 [inlined]
#  [12] (::typeof(∂(mapfoldl_impl)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [13] Pullback (repeats 2 times)
#     @ ./reduce.jl:160 [inlined]
#  [14] (::typeof(∂(mapfoldl)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [15] Pullback
#     @ ./reduce.jl:287 [inlined]
#  [16] (::typeof(∂(#mapreduce#218)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [17] Pullback
#     @ ./reduce.jl:287 [inlined]
#  [18] (::typeof(∂(mapreduce)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [19] Pullback
#     @ ./reduce.jl:501 [inlined]
#  [20] (::typeof(∂(#sum#221)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [21] Pullback
#     @ ./reduce.jl:501 [inlined]
#  [22] (::typeof(∂(sum)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [23] Pullback
#     @ ./reduce.jl:528 [inlined]
#  [24] (::typeof(∂(#sum#222)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [25] Pullback
#     @ ./reduce.jl:528 [inlined]
#  [26] (::typeof(∂(sum)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [27] Pullback
#     @ ~/.julia/dev/DFTK.jl/src/energies.jl:38 [inlined]
#  [28] (::typeof(∂(getproperty)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [29] Pullback
#     @ ~/.julia/packages/ZygoteRules/OjfTt/src/ZygoteRules.jl:11 [inlined]
#  [30] Pullback
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:31 [inlined]
#  [31] (::typeof(∂(compute_energy)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [32] Pullback
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:96 [inlined]
#  [33] (::typeof(∂(#19)))(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface2.jl:0
#  [34] (::Zygote.var"#41#42"{typeof(∂(#19))})(Δ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface.jl:41
#  [35] gradient(f::Function, args::Float64)
#     @ Zygote ~/.julia/packages/Zygote/pM10l/src/compiler/interface.jl:59
#  [36] top-level scope
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:96


using ReverseDiff
ReverseDiff.gradient(a -> compute_energy(scfres, first(a)), [10.26])
# ERROR: LoadError: UndefRefError: access to undefined reference
# Stacktrace:
#   [1] getindex
#     @ ./array.jl:802 [inlined]
#   [2] macro expansion
#     @ ./multidimensional.jl:860 [inlined]
#   [3] macro expansion
#     @ ./cartesian.jl:64 [inlined]
#   [4] macro expansion
#     @ ./multidimensional.jl:855 [inlined]
#   [5] _unsafe_getindex!
#     @ ./multidimensional.jl:868 [inlined]
#   [6] _unsafe_getindex(::IndexLinear, ::Array{Complex{ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}}}, 3}, ::Base.Slice{Base.OneTo{Int64}}, ::Int64, ::Int64)
#     @ Base ./multidimensional.jl:846
#   [7] _getindex
#     @ ./multidimensional.jl:832 [inlined]
#   [8] getindex
#     @ ./abstractarray.jl:1170 [inlined]
#   [9] generic_plan_fft(data::Array{Complex{ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}}}, 3})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/fft_generic.jl:84
#  [10] build_fft_plans(T::Type, fft_size::Tuple{Int64, Int64, Int64})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/fft_generic.jl:41
#  [11] macro expansion
#     @ ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:196 [inlined]
#  [12] (::DFTK.var"#62#64"{ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}}}, Int64})()
#     @ DFTK ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:237
#  [13] timeit(f::DFTK.var"#62#64"{ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}}}, Int64}, to::TimerOutputs.TimerOutput, label::String)
#     @ TimerOutputs ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:285
#  [14] #PlaneWaveBasis#61
#     @ ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:236 [inlined]
#  [15] PlaneWaveBasis(model::Model{ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}}}, Ecut::Int64; kgrid::Vector{Int64}, kshift::Vector{Int64}, use_symmetry::Bool, kwargs::Base.Iterators.Pairs{Symbol, Vector{Int64}, Tuple{Symbol}, NamedTuple{(:fft_size,), Tuple{Vector{Int64}}}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:286
#  [16] make_basis(a::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:12
#  [17] compute_energy(scfres_ref::NamedTuple{(:ham, :basis, :energies, :converged, :ρ, :eigenvalues, :occupation, :εF, :n_iter, :n_ep_extra, :ψ, :diagonalization, :stage), Tuple{Hamiltonian, PlaneWaveBasis{Float64}, Energies{Float64}, Bool, Array{Float64, 4}, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Float64, Int64, Int64, Vector{Matrix{ComplexF64}}, NamedTuple{(:λ, :X, :residual_norms, :iterations, :converged, :n_matvec), Tuple{Vector{Vector{Float64}}, Vector{Matrix{ComplexF64}}, Vector{Vector{Float64}}, Vector{Int64}, Bool, Int64}}, Symbol}}, a::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:19
#  [18] (::var"#19#20")(a::ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:124
#  [19] ReverseDiff.GradientTape(f::var"#19#20", input::Vector{Float64}, cfg::ReverseDiff.GradientConfig{ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}})
#     @ ReverseDiff ~/.julia/packages/ReverseDiff/E4Tzn/src/api/tape.jl:199
#  [20] gradient(f::Function, input::Vector{Float64}, cfg::ReverseDiff.GradientConfig{ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}) (repeats 2 times)
#     @ ReverseDiff ~/.julia/packages/ReverseDiff/E4Tzn/src/api/gradients.jl:22
#  [21] top-level scope
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:124
