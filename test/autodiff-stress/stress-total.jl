# Very basic setup, useful for testing
using DFTK
using Test

a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_atomic(lattice, atoms, symmetries=false)
kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 15          # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

scfres = self_consistent_field(basis, tol=1e-8)

function compute_energy(scfres_ref, a)
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]

    model = model_atomic(lattice, atoms, symmetries=false)
    kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
    Ecut = 15           # kinetic energy cutoff in Hartree
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

    energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=scfres_ref.ρ)
    energies.total
end

compute_energy(scfres, 10.26)

# Finite difference reference stress
FiniteDiff.finite_difference_derivative(a -> compute_energy(scfres, a), 10.26) # -1.4114474091964526

###
### Forward mode
###

using ForwardDiff
ForwardDiff.derivative(a -> compute_energy(scfres, a), 10.26)
# ERROR: LoadError: MethodError: no method matching svdvals!(::Matrix{ForwardDiff.Dual{ForwardDiff.Tag{var"#15#16", Float64}, Float64, 1}})
# Closest candidates are:
#   svdvals!(::SymTridiagonal) at /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/tridiag.jl:351
#   svdvals!(::StridedMatrix{T}) where T<:Union{Float32, Float64, ComplexF32, ComplexF64} at /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/svd.jl:192
#   svdvals!(::StridedMatrix{T}, ::StridedMatrix{T}) where T<:Union{Float32, Float64, ComplexF32, ComplexF64} at /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/svd.jl:498
#   ...
# Stacktrace:
#  [1] svdvals(A::Matrix{ForwardDiff.Dual{ForwardDiff.Tag{var"#15#16", Float64}, Float64, 1}})
#    @ LinearAlgebra /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/svd.jl:217
#  [2] cond(A::Matrix{ForwardDiff.Dual{ForwardDiff.Tag{var"#15#16", Float64}, Float64, 1}}, p::Int64)
#    @ LinearAlgebra /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/dense.jl:1462
#  [3] cond
#    @ /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/dense.jl:1461 [inlined]
#  [4] Model(lattice::Matrix{ForwardDiff.Dual{ForwardDiff.Tag{var"#15#16", Float64}, Float64, 1}}; n_electrons::Nothing, atoms::Vector{Pair{ElementPsp, Vector{Vector{Float64}}}}, magnetic_moments::Vector{Any}, terms::Vector{Any}, temperature::ForwardDiff.Dual{ForwardDiff.Tag{var"#15#16", Float64}, Float64, 1}, smearing::Nothing, spin_polarization::Symbol, symmetries::Bool)
#    @ DFTK ~/.julia/dev/DFTK.jl/src/Model.jl:106
#  [5] model_atomic(lattice::Matrix{ForwardDiff.Dual{ForwardDiff.Tag{var"#15#16", Float64}, Float64, 1}}, atoms::Vector{Pair{ElementPsp, Vector{Vector{Float64}}}}; extra_terms::Vector{Any}, kwargs::Base.Iterators.Pairs{Symbol, Bool, Tuple{Symbol}, NamedTuple{(:symmetries,), Tuple{Bool}}})
#    @ DFTK ~/.julia/dev/DFTK.jl/src/standard_models.jl:20
#  [6] compute_energy(scfres_ref::NamedTuple{(:ham, :basis, :energies, :converged, :ρ, :eigenvalues, :occupation, :εF, :n_iter, :n_ep_extra, :ψ, :diagonalization, :stage), Tuple{Hamiltonian, PlaneWaveBasis{Float64}, Energies{Float64}, Bool, Array{Float64, 4}, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Float64, Int64, Int64, Vector{Matrix{ComplexF64}}, NamedTuple{(:λ, :X, :residual_norms, :iterations, :converged, :n_matvec), Tuple{Vector{Vector{Float64}}, Vector{Matrix{ComplexF64}}, Vector{Vector{Float64}}, Vector{Int64}, Bool, Int64}}, Symbol}}, a::ForwardDiff.Dual{ForwardDiff.Tag{var"#15#16", Float64}, Float64, 1})
#    @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:25
#  [7] (::var"#15#16")(a::ForwardDiff.Dual{ForwardDiff.Tag{var"#15#16", Float64}, Float64, 1})
#    @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:45
#  [8] derivative(f::var"#15#16", x::Float64)
#    @ ForwardDiff ~/.julia/packages/ForwardDiff/m7cm5/src/derivative.jl:14
#  [9] top-level scope
#    @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:45

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
# ERROR: LoadError: MethodError: no method matching svdvals!(::Matrix{ReverseDiff.TrackedReal{Float64, Float64, Nothing}})
# Closest candidates are:
#   svdvals!(::SymTridiagonal) at /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/tridiag.jl:351
#   svdvals!(::StridedMatrix{T}) where T<:Union{Float32, Float64, ComplexF32, ComplexF64} at /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/svd.jl:192
#   svdvals!(::StridedMatrix{T}, ::StridedMatrix{T}) where T<:Union{Float32, Float64, ComplexF32, ComplexF64} at /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/svd.jl:498
#   ...
# Stacktrace:
#   [1] svdvals(A::Matrix{ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}}})
#     @ LinearAlgebra /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/svd.jl:217
#   [2] cond(A::Matrix{ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}}}, p::Int64)
#     @ LinearAlgebra /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/dense.jl:1462
#   [3] cond
#     @ /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/dense.jl:1461 [inlined]
#   [4] Model(lattice::ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}; n_electrons::Nothing, atoms::Vector{Pair{ElementPsp, Vector{Vector{Float64}}}}, magnetic_moments::Vector{Any}, terms::Vector{Any}, temperature::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}}, smearing::Nothing, spin_polarization::Symbol, symmetries::Bool)
#     @ DFTK ~/.julia/dev/DFTK.jl/src/Model.jl:106
#   [5] model_atomic(lattice::ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}, atoms::Vector{Pair{ElementPsp, Vector{Vector{Float64}}}}; extra_terms::Vector{Any}, kwargs::Base.Iterators.Pairs{Symbol, Bool, Tuple{Symbol}, NamedTuple{(:symmetries,), Tuple{Bool}}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/standard_models.jl:20
#   [6] compute_energy(scfres_ref::NamedTuple{(:ham, :basis, :energies, :converged, :ρ, :eigenvalues, :occupation, :εF, :n_iter, :n_ep_extra, :ψ, :diagonalization, :stage), Tuple{Hamiltonian, PlaneWaveBasis{Float64}, Energies{Float64}, Bool, Array{Float64, 4}, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Float64, Int64, Int64, Vector{Matrix{ComplexF64}}, NamedTuple{(:λ, :X, :residual_norms, :iterations, :converged, :n_matvec), Tuple{Vector{Vector{Float64}}, Vector{Matrix{ComplexF64}}, Vector{Vector{Float64}}, Vector{Int64}, Bool, Int64}}, Symbol}}, a::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:25
#   [7] (::var"#23#24")(a::ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:160
#   [8] ReverseDiff.GradientTape(f::var"#23#24", input::Vector{Float64}, cfg::ReverseDiff.GradientConfig{ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}})
#     @ ReverseDiff ~/.julia/packages/ReverseDiff/E4Tzn/src/api/tape.jl:199
#   [9] gradient(f::Function, input::Vector{Float64}, cfg::ReverseDiff.GradientConfig{ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}) (repeats 2 times)
#     @ ReverseDiff ~/.julia/packages/ReverseDiff/E4Tzn/src/api/gradients.jl:22
#  [10] top-level scope
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-total.jl:160

