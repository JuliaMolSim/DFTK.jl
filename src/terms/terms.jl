include("operators.jl")

### Terms
# - A Term is something that, given a state, returns a named tuple (E, hams) with an energy
#   and a list of RFO (for each kpoint).
# - Each term must overload
#     `ene_ops(t, ψ, occ; kwargs...)` -> (E::Real, ops::Vector{RealFourierOperator}).
# - Note that terms are allowed to hold on to references to ψ (eg Fock term),
#   so ψ should not mutated after ene_ops

# The Hamiltonian is defined as half of the gradient of the energy
# with respect to the density matrix sum_n fn |ψn><ψn|.
# In particular, dE/dψn = 2 fn |Hψn> (plus weighting for kpoint sampling)
abstract type Term end

# forces computes either nothing or an array forces[el][at][α]
function forces(term::Term, ψ, occ; kwargs...)
    nothing  # by default, no force
end
function forces(basis::PlaneWaveBasis, ψ, occ; kwargs...)
    # TODO optimize allocs here
    T = eltype(basis)
    f = [zeros(Vec3{T}, length(positions)) for (type, positions) in basis.model.atoms]
    for term in basis.terms
        ft = forces(term, ψ, occ; kwargs...)
        if ft !== nothing
            f += ft
        end
    end
    f
end

"""
A term with a constant zero energy.
"""
struct NoopTerm
    basis::PlaneWaveBasis
end
function ene_ops(term::NoopTerm, ψ, occ; kwargs...)
    (E=zero(real(eltype(ψ[1]))), ops=[NoopRFO(term.basis, kpoint) for kpoint in term.basis.kpoints])
end

include("Hamiltonian.jl")

include("kinetic.jl")
include("local.jl")
include("nonlocal.jl")
include("hartree.jl")
include("power_nonlinearity.jl")
include("xc.jl")
include("ewald.jl")
include("psp_correction.jl")
include("entropy.jl")
include("magnetic.jl")

### Builders are objects X that store the term parameters, and produce a XTerm <: Term when instantiated with a `basis`
