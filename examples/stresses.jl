using DFTK
using Zygote
import FiniteDiff

function make_basis(a)
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp(:Si, functional="lda"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]
    # model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn])
    model = model_DFT(lattice, atoms, [], symmetries=false)
    kgrid = [1, 1, 1]
    Ecut = 7
    PlaneWaveBasis(model, Ecut; kgrid=kgrid)
end

function recompute_energy(a)
    basis = make_basis(a)
    scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-13))
    energies, H = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ)
    energies.total
end

function hellmann_feynman_energy(scfres_ref, a)
    basis = make_basis(a)
    ρ = DFTK.compute_density(basis, scfres_ref.ψ, scfres_ref.occupation)
    energies, H = energy_hamiltonian(basis, scfres_ref.ψ, scfres_ref.occupation; ρ=ρ)
    energies.total
end

a = 10.26
scfres = self_consistent_field(make_basis(a), is_converged=DFTK.ScfConvergenceDensity(1e-13))
hellmann_feynman_energy(a) = hellmann_feynman_energy(scfres, a)

ref_recompute = FiniteDiff.finite_difference_derivative(recompute_energy, a)
ref_hf = FiniteDiff.finite_difference_derivative(hellmann_feynman_energy, a)
s_hf = Zygote.gradient(hellmann_feynman_energy, a)
# ERROR: LoadError: this intrinsic must be compiled to be called
# Stacktrace:
#   [1] macro expansion
#     @ ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0 [inlined]
#   [2] _pullback(::Zygote.Context, ::Core.IntrinsicFunction, ::String, ::Type{Int64}, ::Type{Tuple{Ptr{Int64}}}, ::Ptr{Int64})
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:9
#   [3] _pullback
#     @ ./locks-mt.jl:43 [inlined]
#   [4] _pullback(ctx::Zygote.Context, f::typeof(Base.Threads._get), args::Base.Threads.SpinLock)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#   [5] _pullback
#     @ ./locks-mt.jl:63 [inlined]
#   [6] _pullback(ctx::Zygote.Context, f::typeof(lock), args::Base.Threads.SpinLock)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#   [7] _pullback
#     @ ./condition.jl:73 [inlined]
#   [8] _pullback(ctx::Zygote.Context, f::typeof(lock), args::Base.GenericCondition{Base.Threads.SpinLock})
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#   [9] _pullback
#     @ ./task.jl:305 [inlined]
#  [10] _pullback(::Zygote.Context, ::typeof(Base._wait2), ::Task, ::Task)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#  [11] _pullback
#     @ ./channels.jl:251 [inlined]
#  [12] _pullback
#     @ ./channels.jl:134 [inlined]
#  [13] _pullback(::Zygote.Context, ::Base.var"##_#516", ::Nothing, ::Bool, ::Type{Channel{Tuple{String, Vector{String}, Vector{String}}}}, ::Base.Filesystem.var"#25#28"{String}, ::Int64)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#  [14] _pullback (repeats 2 times)
#     @ ./channels.jl:131 [inlined]
#  [15] _pullback
#     @ ./file.jl:929 [inlined]
#  [16] _pullback(::Zygote.Context, ::Base.Filesystem.var"##walkdir#24", ::Bool, ::Bool, ::typeof(throw), ::typeof(walkdir), ::String)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#  [17] _pullback
#     @ ./file.jl:891 [inlined]
#  [18] _pullback(ctx::Zygote.Context, f::typeof(walkdir), args::String)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#  [19] _pullback
#     @ ~/.julia/dev/DFTK.jl/src/pseudo/list_psp.jl:35 [inlined]
#  [20] _pullback(::Zygote.Context, ::DFTK.var"##list_psp#634", ::String, ::String, ::Symbol, ::String, ::typeof(list_psp), ::Symbol)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#  [21] _pullback
#     @ ~/.julia/dev/DFTK.jl/src/pseudo/list_psp.jl:27 [inlined]
#  [22] _pullback(::Zygote.Context, ::DFTK.var"#list_psp##kw", ::NamedTuple{(:family, :core, :functional), Tuple{String, Symbol, String}}, ::typeof(list_psp), ::Symbol)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#  [23] _pullback
#     @ ~/.julia/dev/DFTK.jl/src/pseudo/load_psp.jl:41 [inlined]
#  [24] _pullback(::Zygote.Context, ::DFTK.var"##load_psp#633", ::String, ::Symbol, ::Base.Iterators.Pairs{Symbol, String, Tuple{Symbol}, NamedTuple{(:functional,), Tuple{String}}}, ::typeof(load_psp), ::Symbol)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#  [25] _pullback
#     @ ~/.julia/dev/DFTK.jl/src/pseudo/load_psp.jl:41 [inlined]
#  [26] _pullback
#     @ ~/.julia/dev/DFTK.jl/examples/stresses.jl:11 [inlined]
#  [27] _pullback(ctx::Zygote.Context, f::typeof(make_basis), args::Float64)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#  [28] _pullback
#     @ ~/.julia/dev/DFTK.jl/examples/stresses.jl:27 [inlined]
#  [29] _pullback(::Zygote.Context, ::typeof(hellmann_feynman_energy), ::NamedTuple{(:ham, :basis, :energies, :converged, :ρ, :eigenvalues, :occupation, :εF, :n_iter, :n_ep_extra, :ψ, :diagonalization, :stage, :algorithm), Tuple{Hamiltonian, PlaneWaveBasis{Float64}, Energies{Float64}, Bool, Array{Float64, 4}, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Float64, Int64, Int64, Vector{Matrix{ComplexF64}}, Vector{NamedTuple{(:λ, :X, :residual_norms, :iterations, :converged, :n_matvec), Tuple{Vector{Vector{Float64}}, Vector{Matrix{ComplexF64}}, Vector{Vector{Float64}}, Vector{Int64}, Bool, Int64}}}, Symbol, String}}, ::Float64)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#  [30] _pullback
#     @ ~/.julia/dev/DFTK.jl/examples/stresses.jl:35 [inlined]
#  [31] _pullback(ctx::Zygote.Context, f::typeof(hellmann_feynman_energy), args::Float64)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface2.jl:0
#  [32] _pullback(f::Function, args::Float64)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface.jl:34
#  [33] pullback(f::Function, args::Float64)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface.jl:40
#  [34] gradient(f::Function, args::Float64)
#     @ Zygote ~/.julia/packages/Zygote/zowrf/src/compiler/interface.jl:58
#  [35] top-level scope
#     @ ~/.julia/dev/DFTK.jl/examples/stresses.jl:39


# isapprox(ref_hf, ref_recompute, atol=1e-4)
# isapprox(s_hf, ref_hf, atol=1e-8)
