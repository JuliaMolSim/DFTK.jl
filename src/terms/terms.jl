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

### Builders are objects X that store the term parameters, and produce a
# XTerm <: Term when instantiated with a `basis`



"""
A term with a constant zero energy.
"""
struct NoopTerm <: Term
    basis::PlaneWaveBasis
end
function ene_ops(term::NoopTerm, ψ, occ; kwargs...)
    (E=zero(eltype(term.basis)), ops=[NoopOperator(term.basis, kpoint)
                                      for kpoint in term.basis.kpoints])
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

# breaks_symmetries on a term builder answers true if this term breaks
# the symmetries of the lattice/atoms (in which case kpoint reduction
# is invalid)
breaks_symmetries(term_builder::Any) = false
breaks_symmetries(term_builder::Magnetic) = true
breaks_symmetries(term_builder::ExternalFromReal) = true
breaks_symmetries(term_builder::ExternalFromFourier) = true


# forces computes either nothing or an array forces[el][at][α]
function forces(term::Term, ψ, occ; kwargs...)
    nothing  # by default, no force
end
@timing function forces(basis::PlaneWaveBasis, ψ, occ; kwargs...)
    if !any(iszero(kpt.coordinate) for kpt in basis.kpoints)
        @warn "Forces for shifted k-Grids not tested"
    end

    # TODO optimize allocs here
    T = eltype(basis)
    f = [zeros(Vec3{T}, length(positions)) for (type, positions) in basis.model.atoms]
    for term in basis.terms
        f_term = forces(term, ψ, occ; kwargs...)
        if !isnothing(f_term)
            f += f_term
        end
    end
    f
end
forces(scfres) = forces(scfres.ham.basis, scfres.ψ, scfres.occupation, ρ=scfres.ρ)


"""
    compute_kernel(term::Term; kwargs...)

Computes a matrix representation of the full response kernel
(derivative of potential with respect to density) in real space.
"""
function compute_kernel(term::Term; kwargs...)
    false  # "strong zero": false + something = something
end
"""
    apply_kernel(term::Term, dρ; kwargs...)

Computes the potential response to a perturbation dρ in real space
"""
function apply_kernel(term::Term, dρ; kwargs...)
    nothing  # by default, no kernel
end

function compute_kernel(basis::PlaneWaveBasis{T}; kwargs...) where {T}
    k = zeros(T, prod(basis.fft_size), prod(basis.fft_size))
    for term in basis.terms
        k .+= compute_kernel(term; kwargs...)
    end
    k
end

function apply_kernel(basis, dρ; RPA=false, kwargs...)
    dV = RealFourierArray(basis)
    for term in basis.terms
        # Skip XC term if RPA is selected
        RPA && term isa TermXc && continue

        dV_term = apply_kernel(term, dρ; kwargs...)
        if !isnothing(dV_term)
            dV.real .+= dV_term.real
        end
    end
    dV
end
