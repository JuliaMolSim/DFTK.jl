using SpecialFunctions: besselj
struct PspTM <: NormConservingPsp
    atomicNumber::Int                               # Atomic number of atom
    valenceElectrons::Int                           # Number of valence electrons
    lmax::Int                                       # Maximal angular momentum in the non-local part
    lloc::Int                                       # Angular momentum used for local part
    numGridPoints::Int                              # Number of grid points used in the radial grid
    r2well::Float64                                 # Prefactor of a harmonic well
    numProjectorFctns::Vector{Int}                  # Number of projector functions used for each angular momentum
    pspCoreRadius::Vector{Float64}                  # Pseudopotential core radius per angular momentum
    rms::Vector{Float64}                            # Measure of pseudopotential quality reflecting the value of the penalty function in designing the potential per angular momentum
    energiesKB::Vector{Vector{Float64}}             # ekb are Kleinman-Bylander energies for each projection function for each angular momentum
    epsatm::Vector{Float64}                         # The integral ∫^∞_0 4πr*(r*V(r)+valenceElectrons)dr
    rchrg::Float64                                  # the Core charge radius for additional core charge used to match the xc contribution
    fchrg::Float64                                  # The prefactor of the core charge expression
    totCoreChrg::Float64                            # The total (integrated) core charge
    PspVals::Vector{Vector{Float64}}                # Radial pseudopotential values for each angular momentum
    firstProjectorVals::Vector{Vector{Float64}}     # Radial First projection functions for each angular momentum
    secondProjectorVals::Vector{Vector{Float64}}    # If numProjectorFctns permits, the second projection functions for each angluar momentum
    identifier::String                              # String identifying the PSP
    description::String                             # Descriptive string
end

function parse_tm_file(path; identifier="")
    file = readlines(path)
    description = file[1]

    Zatom,Zion = Int.(parse.(Float64,split(file[occursin.(r"zatom, zion", file)][1])[1:2]))

    lmax, lloc, numGridPoints, r2well = parse.(Float64,split(file[occursin.(r"lmax,lloc", file)][1])[3:6])

    numProjectorFctns = []; pspCoreRadius = []
    foreach(file[occursin.(r"l,e99.0",file)]) do line
        nproj,rcpsp = parse.(Float64,split(line)[4:5])
        push!(numProjectorFctns,Int(nproj))
        push!(pspCoreRadius,rcpsp)
    end

    rms = []; energiesKB = []; epsatm = []
    foreach(file[occursin.(r"rms,ekb1",file)]) do line
        rmS,ekb1,ekb2,epsatM = parse.(Float64,split(line)[1:4])
        push!(rms,rmS)
        push!(energiesKB,[ekb1,ekb2])
        push!(epsatm,epsatM)
    end

    rchrg, fchrg, totCoreChrg = parse.(Float64,split(file[occursin.(r"rchrg,fchrg", file)][1])[1:3])

    pspVals = map(findall(occursin.(r"for Troullier-Martins",file))) do idx
        parse.(Float64,string.(vcat(split.(file[idx+1:idx+667])...)))
    end

    firstProjectorVals = map(findall(occursin.(r"first projection function",file))) do idx
        parse.(Float64,string.(vcat(split.(file[idx+1:idx+667])...)))
    end

    secondProjectorVals = map(findall(occursin.(r"second projection function",file))) do idx
        parse.(Float64,string.(vcat(split.(file[idx+1:idx+667])...)))
    end

    pspSetupDescription = "\nHere are some more details about setting up of the pseudopotential:\n"
    if isempty(secondProjectorVals)
        if lastindex(file) > findlast(occursin.(r"first projection",file))+667
            description *= pspSetupDescription * prod(line -> line *" \n",file[findlast(occursin.(r"first projection",file))+668 : end])
        end
    else
        if lastindex(file) > findlast(occursin.(r"second projection",file))+667
            description *= pspSetupDescription * prod(line -> line *" \n",file[findlast(occursin.(r"second projection",file))+668 : end])
        end
    end

    function fct(data)
        rng = [100*(x + 0.1)^5 - 1e-8 for x in 0:2000]
        plt = plot()
        foreach(vec -> display(plot!(plt, rng, vec)), data)
    end
    fct(firstProjectorVals)
    PspTM(
        Zatom, Zion,
        Int(lmax), Int(lloc), Int(numGridPoints), r2well,
        numProjectorFctns, pspCoreRadius,
        rms, energiesKB, epsatm,
        rchrg, fchrg, totCoreChrg,
        pspVals,
        firstProjectorVals,
        secondProjectorVals,
        identifier,
        description
    )
end

function eval_psp_energy_correction(psp::PspTM)

end
function eval_psp_local_fourier(psp::PspTM)
    
end

"""
Creating the projector function described in eq. 19 in [the Troullier-Martins](https://doi.org/10.1103/PhysRevB.43.1993) paper
"""
function eval_psp_projection_radial(psp::PspTM, q::T, qPrime::T) where {T <: Real}
    function summations(pseudoWaveFunction,potential,r,q)
        pseudoWaveFunction * potential * besselj(l,abs(q*r))
    end
    nonLocalPotential = ionicPotential - localIonicPotential
    inv(pseudoWaveFunction * (nonLocalPotential * pseudoWaveFunction)) * sum()
end


function eval_psp_local_real(psp::PspTM, q::T) where {T <: Real}
    
end