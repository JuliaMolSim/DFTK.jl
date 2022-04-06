using InteratomicPotentials

# Integration to use DFTK as an interatomic potential
# This is useful for AIMD simulations

struct DFTKPotential <: ArbitraryPotential
    Ecut::Real
    kgrid::AbstractVector{<:Integer}
    scf_args::Dict{Symbol,Any}
    previous_scf::Ref{Any}
end
function DFTKPotential(Ecut::Real, kgrid::AbstractVector{<:Integer}; kwargs...)
    DFTKPotential(Ecut, kgrid, Dict{Symbol,Any}(kwargs...), Ref{Any}())
end
function DFTKPotential(Ecut::Unitful.Energy, kgrid::AbstractVector{<:Integer}; kwargs...)
    DFTKPotential(austrip(Ecut), kgrid, Dict{Symbol,Any}(kwargs...), Ref{Any}())
end

function InteratomicPotentials.energy_and_force(system::AbstractSystem, potential::DFTKPotential)
    model = model_LDA(system)
    basis = PlaneWaveBasis(model; Ecut=potential.Ecut, kgrid=potential.kgrid)

    extra_args = isassigned(potential.previous_scf) ? (ψ=potential.previous_scf[].ψ, ρ=potential.previous_scf[].ρ) : (;)
    scfres = self_consistent_field(basis; potential.scf_args..., extra_args...)
    potential.previous_scf[] = scfres # cache previous scf as starting point for next calculation
    (; e=scfres.energies.total, f=compute_forces_cart(scfres))
end