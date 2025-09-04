using DftFunctionals

include("operators.jl")

# TODO: probably needs Union{..., Nothing} for each field for type stability
@kwdef struct Densities{Tρ, Tτ, Tn}
    ρ::Tρ = nothing
    τ::Tτ = nothing
    hubbard_n::Tn = nothing
end

function sum_densities(a::Densities, b::Densities)
    sum_density(a, b) = isnothing(a) ? b : isnothing(b) ? a : a .+ b

    Densities(sum_density(a.ρ, b.ρ),
              sum_density(a.τ, b.τ),
              sum_density(a.hubbard_n, b.hubbard_n))
end

### Terms
# TODO update docstring
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

# For an orbital term the ops are constant
# and the energy is quadratic in the orbitals and computed from the ops
# It must overload:
# - `ops(term, basis)` returning `Vector{RealFourierOperator}` (for each kpoint)
abstract type OrbitalsTerm <: Term end

# A term that depends on the densities but not on the orbitals
# It must overload:
# - `needed_densities(term)` returning an iterable of symbols
# - `energy_potentials(term, basis, densities::Densities)` returning `(; E, potentials::Densities)`
abstract type DensitiesTerm <: Term end

# TODO: could rename back to TermLinear and TermNonlinear

# Terms that are linear in the density (i.e. they give rise to a Hamiltonian
# which does not dependent on the density, thus they have no kernel)
abstract type LinearDensitiesTerm <: DensitiesTerm end

# Terms that are non-linear in the density (i.e. which give rise to a Hamiltonian
# contribution that is density-dependent as well)
abstract type NonlinearDensitiesTerm <: DensitiesTerm end

compute_kernel(term::LinearDensitiesTerm, basis::AbstractBasis; kwargs...) = nothing
apply_kernel(term::LinearDensitiesTerm, basis::AbstractBasis, δρ; kwargs...) = nothing

### Builders are objects X that store the term parameters, and produce a
# XTerm <: Term when instantiated with a `basis`

# TODO If needed improve this further by specialising energy() for certain terms
function energy(term::Term, basis::AbstractBasis, ψ, occupation; kwargs...)
    ene_ops(term, basis, ψ, occupation; kwargs...).E
end

DftFunctionals.needs_τ(t::Term) = false

"""
A term with a constant zero energy.
"""
struct TermNoop <: LinearDensitiesTerm end
function energy_potentials(term::TermNoop, basis::PlaneWaveBasis{T},
                           densities::Densities) where {T}
    (; E=zero(T), potentials=Densities())
end
needed_densities(::TermNoop) = ()
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
include("hubbard.jl")

include("magnetic.jl")
breaks_symmetries(::Magnetic) = true

include("anyonic.jl")
breaks_symmetries(::Anyonic) = true

# forces computes either nothing or an array forces[at][α] (by default no forces)
compute_forces(::Term, ::AbstractBasis, ψ, occupation; kwargs...) = nothing
# dynamical matrix for phonons computations (array dynmat[3, n_atom, 3, n_atom])
compute_dynmat(::Term, ::AbstractBasis, ψ, occupation; kwargs...) = nothing
# Get δH ψ, with δH the perturbation of the Hamiltonian due to a position displacement
# e^{iq·r} of the α coordinate of atom s.
# δHψ[ik] is δH ψ_{k-q}, expressed in basis.kpoints[ik].
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
