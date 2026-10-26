module DFTKSimpleDFTD3Ext
# Only `import DFTK`, such that the builder below can have the same name as `DFTK.DFTD3`,
# which is also the name under which the energy is reported.
import DFTK
using DFTK: PlaneWaveBasis, TermLinear, NoopOperator, Vec3, charge_nuclear
using ForwardDiff
using LinearAlgebra
using SimpleDFTD3

struct DFTD3
    damping  # A SimpleDFTD3 damping or `nothing` for D3(BJ) of the model's functional
    cutoffs::NamedTuple
end
DFTK.DFTD3(damping=nothing; cutoffs=(;)) = DFTD3(damping, cutoffs)

function Base.show(io::IO, d3::DFTD3)
    print(io, "DFTD3(")
    isnothing(d3.damping) || print(io, dispersion_identifier(d3.damping))
    isempty(d3.cutoffs) || print(io, "; cutoffs=", d3.cutoffs)
    print(io, ")")
end

function _default_damping(model)
    xcs = [term for term in model.term_types if term isa DFTK.Xc]
    isempty(xcs) && throw(ArgumentError("The model has no exchange-correlation " *
                                        "functional, so DFTD3 needs an explicit damping."))
    xc = only(xcs)
    identifiers = DFTK.DftFunctionals.identifier.(xc.functionals)

    # The stored D3 parameters are fitted to the unmodified functional, so the Xc term must
    # not be scaled and the exact exchange must be the one of the functional (if any)
    is_exx(term) = term isa DFTK.ExactExchange
    exx = filter(is_exx, model.term_types)
    exx_default = isnothing(DFTK.hybrid_parameters(xc)) ? [] :
                  filter(is_exx, DFTK.HybridFunctional(identifiers))
    if !isone(xc.scaling_factor) || exx != exx_default
        throw(ArgumentError("The exchange-correlation functional of the model is " *
                            "modified, so DFTD3 needs an explicit damping."))
    end
    RationalDamping(dftd3_method_name(identifiers))
end

DFTK.@timing "precomp: DFTD3" function (d3::DFTD3)(basis::PlaneWaveBasis{T}) where {T}
    model = basis.model
    model.n_dim == 3 ||
        throw(ArgumentError("DFTD3 is only implemented for fully periodic 3D systems."))
    numbers = charge_nuclear.(model.atoms)
    # simple-dftd3 has no data beyond Z = 103 and reads out of bounds there
    all(Z -> 1 ≤ Z ≤ 103, numbers) ||
        throw(ArgumentError("DFTD3 is only available for atomic numbers 1 to 103."))
    damping = isnothing(d3.damping) ? _default_damping(model) : d3.damping

    # simple-dftd3 only takes plain numbers, so evaluate at the values of any dual numbers
    L0 = ForwardDiff.value.(model.lattice)
    x0 = [ForwardDiff.value.(x) for x in model.positions]
    # Nested dual numbers ask for second derivatives, which the first-order expansion below
    # does not have
    eltype(L0) <: ForwardDiff.Dual && error("Second derivatives of the DFTD3 energy " *
                                            "(e.g. elastic constants) are not implemented.")
    dispersion_model = DispersionModel(numbers, L0 * stack(x0); lattice=L0)
    set_realspace_cutoff!(dispersion_model; d3.cutoffs...)
    (; energy, gradient, sigma) = dispersion(dispersion_model, damping)

    # sigma is the strain derivative (L → (1+ε)L at fixed fractional positions), so
    # dE/dL = sigma L⁻ᵀ, and for fractional positions dE/dx = Lᵀ g with g the Cartesian
    # gradient. Adding these as a first-order expansion around the values gives correct
    # ForwardDiff derivatives, e.g. for stresses.
    dE_dL = sigma / L0'
    dE_dx = [L0' * g for g in eachcol(gradient)]
    ΔL = model.lattice - L0
    Δx = model.positions - x0
    E  = energy + dot(dE_dL, ΔL) + sum(dot.(dE_dx, Δx); init=0)
    TermDFTD3{T}(E, [Vec3{T}(-g) for g in dE_dx])
end

struct TermDFTD3{T} <: TermLinear
    energy::T
    forces::Vector{Vec3{T}}
end

function DFTK.ene_ops(term::TermDFTD3, basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    (; E=term.energy, ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
end
DFTK.compute_forces(term::TermDFTD3, ::PlaneWaveBasis, ψ, occ; kwargs...) = term.forces
# The first-order expansion above gives derivatives of the energy, but not of the forces
function DFTK.compute_forces(::TermDFTD3{<:ForwardDiff.Dual}, ::PlaneWaveBasis,
                             ψ, occupation; kwargs...)
    error("Derivatives of the DFTD3 forces are not implemented.")
end
function DFTK.compute_dynmat(::TermDFTD3, ::PlaneWaveBasis, ψ, occupation; kwargs...)
    error("Phonons are not implemented for DFTD3.")
end

end
