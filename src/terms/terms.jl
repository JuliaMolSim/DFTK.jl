include("operators.jl")

### Terms
# - A Term is something that, given a state, returns a named tuple (; E, hams) with an energy
#   and a list of RealFourierOperator (for each kpoint).
# - Each term must overload
#     `ene_ops(term, basis, ψ, occupation; kwargs...)`
#         -> (; E::Real, ops::Vector{RealFourierOperator}).
# - Note that terms are allowed to hold on to references to ψ (eg Fock term),
#   so ψ should not mutated after ene_ops

# The Hamiltonian is defined as half of the gradient of the energy
# with respect to the density matrix sum_n fn |ψn><ψn|.
# In particular, dE/dψn = 2 fn |Hψn> (plus weighting for k-point sampling)
abstract type Term end

# Terms that are non-linear in the density (i.e. which give rise to a Hamiltonian
# contribution that is density-dependent or orbital-dependent as well)
abstract type TermNonlinear <: Term end

### Builders are objects X that store the term parameters, and produce a
# XTerm <: Term when instantiated with a `basis`


"""
A term with a constant zero energy.
"""
struct TermNoop <: Term end
function ene_ops(term::TermNoop, basis::PlaneWaveBasis{T}, ψ, occupation; kwargs...) where {T}
    (; E=zero(eltype(T)), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
end

include("Hamiltonian.jl")

# breaks_symmetries on a term builder answers true if this term breaks
# the symmetries of the lattice/atoms (in which case k-point reduction
# is invalid)
breaks_symmetries(::Any) = false

include("kinetic.jl")

include("local.jl")
breaks_symmetries(::ExternalFromReal) = true
breaks_symmetries(::ExternalFromFourier) = true

include("nonlocal.jl")
include("hartree.jl")
include("local_nonlinearity.jl")
include("xc.jl")
include("ewald.jl")
include("psp_correction.jl")
include("entropy.jl")
include("pairwise.jl")

include("magnetic.jl")
breaks_symmetries(::Magnetic) = true

include("anyonic.jl")
breaks_symmetries(::Anyonic) = true

# forces computes either nothing or an array forces[at][α] (by default no forces)
compute_forces(::Term, ::AbstractBasis, ψ, occupation; kwargs...) = nothing
# dynamical matrix for phonons computations (array dynmat[3, n_atom, 3, n_atom])
compute_dynmat(::Term, ::AbstractBasis, ψ, occupation; kwargs...) = nothing
# variation of the Hamiltonian applied to orbitals for phonons computations
compute_δHψ_αs(::Term, ::AbstractBasis, ψ, α, s, q; kwargs...) = nothing

@doc raw"""
    compute_kernel(basis::PlaneWaveBasis; kwargs...)

Computes a matrix representation of the full response kernel
(derivative of potential with respect to density) in real space.
For non-spin-polarized calculations the matrix dimension is
`prod(basis.fft_size)` × `prod(basis.fft_size)` and
for collinear spin-polarized cases it is
`2prod(basis.fft_size)` × `2prod(basis.fft_size)`.
In this case the matrix has effectively 4 blocks
```math
\left(\begin{array}{cc}
    K_{αα} & K_{αβ}\\
    K_{βα} & K_{ββ}
\end{array}\right)
```
"""
@timing function compute_kernel(basis::PlaneWaveBasis{T}; kwargs...) where {T}
    n_spin = basis.model.n_spin_components
    kernel = zeros(T, n_spin * prod(basis.fft_size), n_spin * prod(basis.fft_size))
    for term in basis.terms
        isnothing(term) && continue
        kernel .+= compute_kernel(term, basis; kwargs...)
    end
    kernel
end
compute_kernel(::Term, ::AbstractBasis{T}; kwargs...) where {T} = nothing  # By default no kernel


"""
    apply_kernel(basis::PlaneWaveBasis, δρ; kwargs...)

Computes the potential response to a perturbation δρ in real space,
as a 4D `(i,j,k,σ)` array.
"""
@timing function apply_kernel(basis::PlaneWaveBasis, δρ; RPA=false, kwargs...)
    n_spin = basis.model.n_spin_components
    @assert 1 ≤ n_spin ≤ 2

    if RPA
        hartree = filter(t -> t isa TermHartree, basis.terms)
        δV = isempty(hartree) ? zero(δρ) : apply_kernel(only(hartree), basis, δρ; kwargs...)
    else
        δV = zero(δρ)
        for term in basis.terms
            δV_term = apply_kernel(term, basis, δρ; kwargs...)
            if !isnothing(δV_term)
                δV .+= δV_term
            end
        end
    end
    δV
end
apply_kernel(::Term, ::AbstractBasis{T}, δρ; kwargs...) where {T} = nothing  # by default, no kernel
