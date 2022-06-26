module Aiida  # Special submodule for Aiida integration

# TODO This should be integrated with the interatomic potentials stuff

# Error handling:
#    - unit tests for std error messages
#
# Geometry optimisation
# Equation of state

@kwdef struct DftkParameters
    system::AbstractSystem
    functionals::Vector{Symbol}  # libxc convention
    #
    temperature::Float64 = 0.0
    smearing::Dict = Dict(; name=:None)
    #
    Ecut::Float64
    kgrid::Tuple{Int,Int,Int}
    kshift::Tuple{Float64,Float64,Float64}
    #
    tolscf::Float64  # in density
    damping::Float64 = 0.8
    mixing::Dict = Dict(; name=:LdosMixing)
    maxiter::Int = 100
    #
    outfile::String  # Output file name prefix
end
# TODO Go for HDF5 output

function PlaneWaveBasis(params::DftkParameters)
    smearing = let
        smearing_kwargs = copy(params.smearing)
        smearing_name   = pop!(smearing_kwargs, :name)
        if !(smearing_name in (:None, :FermiDirac, :Gaussian, :MethfesselPaxton, :MarzariVanderbilt))
            error("Unknown smearing: $smearing_name")
        end
        getproperty(DFTK.Smearing, smearing_name)(; smearing_kwargs...)
    end

    model    = model_DFT(params.system, params.functionals;
                         params.temperature, smearing)
    PlaneWaveBasis(model; params.Ecut, params.kgrid, params.kshift)
end



function ground_state(params::DftkParameters)
    basis = PlaneWaveBasis(params)

    mixing = let
        mixing_kwargs = copy(params.mixing)
        mixing_name   = pop!(mixing_kwargs)
        if !(calc.mixing in (:LdosMixing, :KerkerMixing, :SimpleMixing,
                             :KerkerDosMixing, :DielectricMixing))
            error("Unknown mixing: $(calc.mixing)")
        end
        getproperty(DFTK, calc.mixing)(; mixing_kwargs...)
    end

    scfres   = self_consistent_field(basis; calc.damping, calc.maxiter, mixing,
                                     ρ=guess_density(basis, calc.system),
                                     is_converged=DFTK.ScfConvergenceDensity(calc.tolscf))
    save_scfres(scfres, calc.outfile * ".jld2")
    nothing
end

# TODO That should not stay in DFTK ... much rather there should be a generic calculatior
#      interface and this uses it.
function geometry_optimisation(calc::DftkParameters)
    scfres = load_scfres(calc.outfile * ".jld2")


    new_basis = PlaneWaveBasis(basis, new_positions)


end

end
