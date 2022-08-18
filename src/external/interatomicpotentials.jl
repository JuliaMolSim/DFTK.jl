# import InteratomicPotentials: NonTrainablePotential, energy_and_force

# # Integration to use DFTK as an interatomic potential
# # This is useful for AIMD simulations

# Base.@kwdef struct DFTKPotential <: NonTrainablePotential
#     functionals::Vector{Symbol}    = [:gga_x_pbe, :gga_c_pbe]  # default to model_PBE
#     model_kwargs::Dict{Symbol,Any} = Dict{Symbol,Any}()
#     basis_kwargs::Dict{Symbol,Any} = Dict{Symbol,Any}()
#     scf_kwargs::Dict{Symbol,Any}   = Dict{Symbol,Any}()
# end
# function DFTKPotential(Ecut, kgrid; kwargs...)
#     p = DFTKPotential(; kwargs...)
#     p.basis_kwargs[:Ecut]  = Ecut
#     p.basis_kwargs[:kgrid] = kgrid
#     p
# end

# function energy_and_force(system::AbstractSystem, potential::DFTKPotential)
#     model = model_DFT(system, potential.functionals; potential.model_kwargs...)
#     basis = PlaneWaveBasis(model; potential.basis_kwargs...)

#     scfres = self_consistent_field(basis; potential.scf_kwargs...)
#     # cache ψ and ρ as starting point for next calculation
#     potential.scf_kwargs[:ψ] = scfres.ψ
#     potential.scf_kwargs[:ρ] = scfres.ρ

#     (; e=scfres.energies.total * u"hartree", f=compute_forces_cart(scfres) * u"hartree/bohr")
# end
