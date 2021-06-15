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
ForwardDiff.derivative(compute_energy, 10.26) # -2.948556665529993e9

#===#
# selected previous stack traces below.

# ERROR: LoadError: MethodError: no method matching mul!(
#    ::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true}, 
#    ::FFTW.cFFTWPlan{ComplexF64, 1, false, 3, UnitRange{Int64}}, 
#    ::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true}, 
#    ::Bool, 
#    ::Bool
# )
# Closest candidates are:
#   mul!(::AbstractArray, ::Number, ::AbstractArray, ::Number, ::Number) at /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/generic.jl:132
#   mul!(::AbstractArray, ::AbstractArray, ::Number, ::Number, ::Number) at /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/generic.jl:140
#   mul!(::Any, ::Any, ::Any) at /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/matmul.jl:274
#   ...
# Stacktrace:
#   [1] mul!
#     @ /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/matmul.jl:275 [inlined]
#   [2] mul!(y::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true}, p::AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, false, 3, UnitRange{Int64}}, ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, x::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true})
#     @ AbstractFFTs ~/.julia/packages/AbstractFFTs/JebmH/src/definitions.jl:269
#   [3] G_to_r!(f_real::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true}, basis::PlaneWaveBasis{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, f_fourier::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:383
#   [4] G_to_r(basis::PlaneWaveBasis{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, f_fourier::Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3}; assume_real::Bool)
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:414
#   [5] G_to_r
#     @ ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:410 [inlined]
#   [6] (::AtomicLocal)(basis::PlaneWaveBasis{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/terms/local.jl:93
#   [7] macro expansion
#     @ ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:190 [inlined]
#   [8] macro expansion
#     @ ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:246 [inlined]
#   [9] (::DFTK.var"#66#68"{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Int64})()
#     @ DFTK ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:237
#  [10] timeit(f::DFTK.var"#66#68"{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Int64}, to::TimerOutputs.TimerOutput, label::String)
#     @ TimerOutputs ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:285
#  [11] #PlaneWaveBasis#65
#     @ ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:236 [inlined]
#  [12] PlaneWaveBasis(model::Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Ecut::Int64; kgrid::Vector{Int64}, kshift::Vector{Int64}, use_symmetry::Bool, kwargs::Base.Iterators.Pairs{Symbol, Vector{Int64}, Tuple{Symbol}, NamedTuple{(:fft_size,), Tuple{Vector{Int64}}}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:286
#  [13] make_basis(a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:19
#  [14] compute_energy(scfres_ref::NamedTuple{(:ham, :basis, :energies, :converged, :ρ, :eigenvalues, :occupation, :εF, :n_iter, :n_ep_extra, :ψ, :diagonalization, :stage), Tuple{Hamiltonian, PlaneWaveBasis{Float64}, Energies{Float64}, Bool, Array{Float64, 4}, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Float64, Int64, Int64, Vector{Matrix{ComplexF64}}, NamedTuple{(:λ, :X, :residual_norms, :iterations, :converged, :n_matvec), Tuple{Vector{Vector{Float64}}, Vector{Matrix{ComplexF64}}, Vector{Vector{Float64}}, Vector{Int64}, Bool, Int64}}, Symbol}}, a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:29
#  [15] compute_energy(a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:34
#  [16] derivative(f::typeof(compute_energy), x::Float64)
#     @ ForwardDiff ~/.julia/packages/ForwardDiff/QOqCN/src/derivative.jl:14
#  [17] top-level scope
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:41



# ERROR: LoadError: MethodError: mul!(::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true}, ::AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, false, 3, UnitRange{Int64}}, ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, ::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true}) is ambiguous. Candidates:
#   mul!(Y, p::AbstractFFTs.ScaledPlan, X::AbstractArray{var"#s25", N} where {var"#s25"<:(Complex{var"#s24"} where var"#s24"<:ForwardDiff.Dual), N}) in DFTK at /home/niku/.julia/dev/DFTK.jl/src/fft.jl:241
#   mul!(Y, p::AbstractFFTs.Plan, X::AbstractArray{var"#s25", N} where {var"#s25"<:(Complex{var"#s24"} where var"#s24"<:ForwardDiff.Dual), N}) in DFTK at /home/niku/.julia/dev/DFTK.jl/src/fft.jl:241
#   mul!(y::AbstractArray, p::AbstractFFTs.ScaledPlan, x::AbstractArray) in AbstractFFTs at /home/niku/.julia/packages/AbstractFFTs/JebmH/src/definitions.jl:269
# Possible fix, define
#   mul!(::AbstractArray, ::AbstractFFTs.ScaledPlan, ::AbstractArray{var"#s25", N} where {var"#s25"<:(Complex{var"#s24"} where var"#s24"<:ForwardDiff.Dual), N})
# Stacktrace:
#   [1] G_to_r!(f_real::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true}, basis::PlaneWaveBasis{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, f_fourier::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:383
#   [2] G_to_r(basis::PlaneWaveBasis{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, f_fourier::Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3}; assume_real::Bool)
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:414
#   [3] G_to_r
#     @ ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:410 [inlined]
#   [4] (::AtomicLocal)(basis::PlaneWaveBasis{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/terms/local.jl:93
#   [5] macro expansion
#     @ ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:190 [inlined]
#   [6] macro expansion
#     @ ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:246 [inlined]
#   [7] (::DFTK.var"#77#79"{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Int64})()
#     @ DFTK ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:237
#   [8] timeit(f::DFTK.var"#77#79"{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Int64}, to::TimerOutputs.TimerOutput, label::String)
#     @ TimerOutputs ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:285
#   [9] #PlaneWaveBasis#76
#     @ ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:236 [inlined]
#  [10] PlaneWaveBasis(model::Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Ecut::Int64; kgrid::Vector{Int64}, kshift::Vector{Int64}, use_symmetry::Bool, kwargs::Base.Iterators.Pairs{Symbol, Vector{Int64}, Tuple{Symbol}, NamedTuple{(:fft_size,), Tuple{Vector{Int64}}}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:286
#  [11] make_basis(a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:19
#  [12] compute_energy(scfres_ref::NamedTuple{(:ham, :basis, :energies, :converged, :ρ, :eigenvalues, :occupation, :εF, :n_iter, :n_ep_extra, :ψ, :diagonalization, :stage), Tuple{Hamiltonian, PlaneWaveBasis{Float64}, Energies{Float64}, Bool, Array{Float64, 4}, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Float64, Int64, Int64, Vector{Matrix{ComplexF64}}, NamedTuple{(:λ, :X, :residual_norms, :iterations, :converged, :n_matvec), Tuple{Vector{Vector{Float64}}, Vector{Matrix{ComplexF64}}, Vector{Vector{Float64}}, Vector{Int64}, Bool, Int64}}, Symbol}}, a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:29
#  [13] compute_energy(a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:34
#  [14] derivative(f::typeof(compute_energy), x::Float64)
#     @ ForwardDiff ~/.julia/packages/ForwardDiff/QOqCN/src/derivative.jl:14
#  [15] top-level scope
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:41



# ERROR: LoadError: MethodError: no method matching Float64(::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
# Closest candidates are:
#   (::Type{T})(::Real, ::RoundingMode) where T<:AbstractFloat at rounding.jl:200
#   (::Type{T})(::T) where T<:Number at boot.jl:760
#   (::Type{T})(::AbstractChar) where T<:Union{AbstractChar, Number} at char.jl:50
#   ...
# Stacktrace:
#   [1] convert(#unused#::Type{Float64}, x::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Base ./number.jl:7
#   [2] ComplexF64(re::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}, im::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Base ./complex.jl:12
#   [3] ComplexF64(z::Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}})
#     @ Base ./complex.jl:36
#   [4] convert(#unused#::Type{ComplexF64}, x::Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}})
#     @ Base ./number.jl:7
#   [5] setindex!(A::Array{ComplexF64, 3}, x::Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, i1::Int64)
#     @ Base ./array.jl:839
#   [6] macro expansion
#     @ /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/generic.jl:183 [inlined]
#   [7] macro expansion
#     @ ./simdloop.jl:77 [inlined]
#   [8] rmul!
#     @ /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.6/LinearAlgebra/src/generic.jl:182 [inlined]
#   [9] *(p::AbstractFFTs.ScaledPlan{
#           ComplexF64, 
#           FFTW.cFFTWPlan{ComplexF64, 1, false, 3, UnitRange{Int64}}, 
#           ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}
#         }, 
#         x::Array{ComplexF64, 3})
#     @ AbstractFFTs ~/.julia/packages/AbstractFFTs/JebmH/src/definitions.jl:249
#  [10] _apply_plan(p::AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, false, 3, UnitRange{Int64}}, ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, x::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/fft.jl:249
#  [11] mul!(Y::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true}, p::AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, false, 3, UnitRange{Int64}}, ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, X::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/fft.jl:243
#  [12] G_to_r!(f_real::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true}, basis::PlaneWaveBasis{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, f_fourier::SubArray{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3, Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 4}, Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:383
#  [13] G_to_r(basis::PlaneWaveBasis{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, f_fourier::Array{Complex{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, 3}; assume_real::Bool)
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:414
#  [14] G_to_r
#     @ ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:410 [inlined]
#  [15] (::AtomicLocal)(basis::PlaneWaveBasis{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/terms/local.jl:93
#  [16] macro expansion
#     @ ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:190 [inlined]
#  [17] macro expansion
#     @ ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:246 [inlined]
#  [18] (::DFTK.var"#77#79"{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Int64})()
#     @ DFTK ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:237
#  [19] timeit(f::DFTK.var"#77#79"{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}, Bool, Bool, Int64, Vector{Int64}, Vector{Int64}, Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Int64}, to::TimerOutputs.TimerOutput, label::String)
#     @ TimerOutputs ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:285
#  [20] #PlaneWaveBasis#76
#     @ ~/.julia/packages/TimerOutputs/ZmKD7/src/TimerOutput.jl:236 [inlined]
#  [21] PlaneWaveBasis(model::Model{ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1}}, Ecut::Int64; kgrid::Vector{Int64}, kshift::Vector{Int64}, use_symmetry::Bool, kwargs::Base.Iterators.Pairs{Symbol, Vector{Int64}, Tuple{Symbol}, NamedTuple{(:fft_size,), Tuple{Vector{Int64}}}})
#     @ DFTK ~/.julia/dev/DFTK.jl/src/PlaneWaveBasis.jl:286
#  [22] make_basis(a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:19
#  [23] compute_energy(scfres_ref::NamedTuple{(:ham, :basis, :energies, :converged, :ρ, :eigenvalues, :occupation, :εF, :n_iter, :n_ep_extra, :ψ, :diagonalization, :stage), Tuple{Hamiltonian, PlaneWaveBasis{Float64}, Energies{Float64}, Bool, Array{Float64, 4}, Vector{Vector{Float64}}, Vector{Vector{Float64}}, Float64, Int64, Int64, Vector{Matrix{ComplexF64}}, NamedTuple{(:λ, :X, :residual_norms, :iterations, :converged, :n_matvec), Tuple{Vector{Vector{Float64}}, Vector{Matrix{ComplexF64}}, Vector{Vector{Float64}}, Vector{Int64}, Bool, Int64}}, Symbol}}, a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:29
#  [24] compute_energy(a::ForwardDiff.Dual{ForwardDiff.Tag{typeof(compute_energy), Float64}, Float64, 1})
#     @ Main ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:34
#  [25] derivative(f::typeof(compute_energy), x::Float64)
#     @ ForwardDiff ~/.julia/packages/ForwardDiff/QOqCN/src/derivative.jl:14
#  [26] top-level scope
#     @ ~/.julia/dev/DFTK.jl/test/autodiff-stress/stress-forward.jl:41
