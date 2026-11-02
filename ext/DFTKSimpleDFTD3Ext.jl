module DFTKSimpleDFTD3Ext
# Only `import DFTK`, such that the builder below can have the same name as `DFTK.DFTD3`,
# which is also the name under which the energy is reported.
import DFTK
using DFTK: PlaneWaveBasis, TermLinear, NoopOperator, Element, Vec3, charge_nuclear
using ForwardDiff
using LinearAlgebra
using SimpleDFTD3

function SimpleDFTD3.dftd3_method_name(xc::DFTK.Xc)
    dftd3_method_name(DFTK.DftFunctionals.identifier.(xc.functionals))
end

function SimpleDFTD3.DispersionModel(lattice::AbstractMatrix,
                                     atoms::AbstractVector{<:Element},
                                     positions::AbstractVector{<:AbstractVector})
    numbers = charge_nuclear.(atoms)
    all(Z -> 1 ≤ Z ≤ 103, numbers) ||
        throw(ArgumentError("DFTD3 is only available for atomic numbers 1 to 103."))
    L = ForwardDiff.value.(lattice)  # Remove duals as simple-dftd3 only takes plain numbers.
    DispersionModel(numbers, L * stack(x -> ForwardDiff.value.(x), positions); lattice=L)
end

struct DFTD3
    damping::Union{String,AbstractDamping}  # Identifier (e.g. "D3(BJ)") or a damping
    method::Union{Nothing,String}           # Only used if string identifier given above
    three_body::Bool                        # Only used if string identifier given above
    dispersion_model::Union{Nothing,DispersionModel}
    switching_width_2body::Float64          # Only used without a dispersion_model
    switching_width_3body::Float64
end
function DFTK.DFTD3(method::AbstractString; damping::AbstractString="D3(BJ)",
                    three_body=false, switching_width_2body=0.0, switching_width_3body=0.0)
    DFTD3(damping, method, three_body, nothing, switching_width_2body, switching_width_3body)
end
function DFTK.DFTD3(method::AbstractString, dispersion_model::DispersionModel;
                    damping::AbstractString="D3(BJ)", three_body=false)
    DFTD3(damping, method, three_body, dispersion_model, 0.0, 0.0)
end
function DFTK.DFTD3(damping::AbstractDamping; switching_width_2body=0.0, switching_width_3body=0.0)
    DFTD3(damping, nothing, false, nothing, switching_width_2body, switching_width_3body)
end
function DFTK.DFTD3(damping::AbstractDamping, dispersion_model::DispersionModel)
    DFTD3(damping, nothing, false, dispersion_model, 0.0, 0.0)
end
function DFTK.DFTD3(xc::DFTK.Xc, args...; kwargs...)
    DFTK.DFTD3(dftd3_method_name(xc), args...; kwargs...)
end
function DFTK.DFTD3(functionals::AbstractVector, args...; kwargs...)
    xc = first(DFTK._parse_functionals(functionals).dftterms)
    DFTK.DFTD3(xc, args...; kwargs...)
end

function Base.show(io::IO, d3::DFTD3)
    if d3.damping isa AbstractDamping || !isnothing(d3.dispersion_model)
        return print(io, "DFTD3(custom)")
    end
    print(io, "DFTD3(", repr(d3.method), "; damping=$(repr(d3.damping))")
    print(io, ", three_body=$(d3.three_body)")
    d3.switching_width_2body > 0 &&
        print(io, ", switching_width_2body=$(d3.switching_width_2body)")
    d3.switching_width_3body > 0 &&
        print(io, ", switching_width_3body=$(d3.switching_width_3body)")
    print(io, ")")
end

DFTK.@timing "precomp: DFTD3" function (d3::DFTD3)(basis::PlaneWaveBasis{T}) where {T}
    model = basis.model
    if model.n_dim != 3
        throw(ArgumentError("DFTD3 is only implemented for fully periodic 3D systems."))
    end
    L0 = ForwardDiff.value.(model.lattice)
    x0 = [ForwardDiff.value.(x) for x in model.positions]
    @assert !(eltype(L0) <: ForwardDiff.Dual)  # i.e. not computing a second derivative

    if isnothing(d3.dispersion_model)
        dispersion_model = DispersionModel(L0, model.atoms, x0)
        set_realspace_cutoff!(dispersion_model; width2=d3.switching_width_2body,
                                                width3=d3.switching_width_3body)
    else
        dispersion_model = d3.dispersion_model
    end
    if d3.damping isa String
        damping = Damping(d3.damping, d3.method; atm=d3.three_body)
    else
        damping = d3.damping
    end
    (; energy, gradient, sigma) = dispersion(dispersion_model, damping)

    # sigma is the strain derivative (L → (1+ε)L at fixed fractional positions), so
    # dE/dL = sigma L⁻ᵀ, and for fractional positions dE/dx = Lᵀ g with g the Cartesian
    # gradient. Adding these as a first-order expansion around the values gives correct
    # ForwardDiff derivatives, e.g. for stresses.
    dE_dL = sigma / L0'
    dE_dx = [L0' * g for g in eachcol(gradient)]
    δL = model.lattice - L0
    δx = model.positions - x0
    E  = energy + dot(dE_dL, δL) + sum(dot.(dE_dx, δx); init=0)
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

function DFTK.compute_forces(::TermDFTD3{<:ForwardDiff.Dual}, ::PlaneWaveBasis,
                             ψ, occupation; kwargs...)
    error("Derivatives of the DFTD3 forces are not implemented.")
end
function DFTK.compute_dynmat(::TermDFTD3, ::PlaneWaveBasis, ψ, occupation; kwargs...)
    error("Phonons are not implemented for DFTD3.")
end

end
