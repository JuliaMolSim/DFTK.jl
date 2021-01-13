using SpecialFunctions: besselj
using Interpolations: interpolate, Gridded, Linear
using Plots: plot, plot!
using Legendre: Plm

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
    pseudoPotentials::Vector{Vector{Float64}}       # Radial pseudopotential values for each angular momentum
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

    pseudoPotentials = map(findall(occursin.(r"for Troullier-Martins",file))) do idx
        parse.(Float64,string.(vcat(split.(file[idx+1:idx+667])...)))
    end

    firstProjectorVals = map(findall(occursin.(r"first projection function",file))) do idx
        parse.(Float64,string.(vcat(split.(file[idx+1:idx+667])...)))
    end
    xval = [100*((x/2000)+0.1)^5-1e-8 for x in 0:2000]
    firstProjectorFunctions = map(projectorValues -> interpolate((xval,), projectorValues, Gridded(Linear())), firstProjectorVals)
    @show typeof(firstProjectorFunctions[1])

    secondProjectorVals = map(findall(occursin.(r"second projection function",file))) do idx
        parse.(Float64,string.(vcat(split.(file[idx+1:idx+667])...)))
    end
    isempty(secondProjectorVals) || (secondProjectorFunctions = map(projVals -> interpolate((xval,), projVals, Gridded(Linear())), secondProjectorVals))

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
        xvals = [100*((x/2000)+0.1)^5-1e-8 for x in 0:2000]
        plt = plot()
        foreach(rge -> display(plot!(plt,xvals,rge)), data[1:1])
    end

    # fct(firstProjectorVals)

    PspTM(
        Zatom, Zion,
        Int(lmax), Int(lloc), Int(numGridPoints), r2well,
        numProjectorFctns, pspCoreRadius,
        rms, energiesKB, epsatm,
        rchrg, fchrg, totCoreChrg,
        pseudoPotentials,
        firstProjectorVals,
        secondProjectorVals,
        identifier,
        description
    )
end


function eval_psp_energy_correction(psp::PspTM,r::T) where {T <: Real} #From https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90 Line: 658 in the psp1nl function
    radialGrid = [100*((x/2000)+0.1)^5-1e-8 for x in 0:2000]
    pseudoWaveFunction = interpolate((radialGrid,), psp.firstProjectorVals[anglularMomentum + 1], Gridded(Linear()))
    integrate(pseudoWaveFunction * (eval_psp_semilocal_real^2 * pseudoWaveFunction), 0, Inf)
end

function eval_psp_local_fourier(psp::PspTM), q::T, qPrime::T, angularMomentum::Int, localAngularMom::Int) where {T <: Real}
    r = 2*π/magnitude(q)
    magnitude(vector) = sum(√(a.^2))
    γ = dot(q,qPrime)/(magnitude(q) * magnitude(qPrime))
    legendre = (2 * angularMomentum + 1)/(4*π) * Plm(angularMomentum,1, cos(γ))
    integrate(eval_psp_semilocal_real(psp,r,angularMomentum, localAngularMom) * besselj(angularMomentum,magnitude(q)) * besselj(angularMomentum, magnitude(qPrime)) * r^2, 0, Inf)
end

"""
This is just creating a function from the pseudopotential that came from the file.
"""
function eval_psp_local_real(psp::PspTM, r::T, angularMomentum::Int) where {T <: Real}
    radialGrid = [100*((x/2000)+0.1)^5-1e-8 for x in 0:2000]
    interpolate((radialGrid,), psp.pseudoPotentials[angularMomentum + 1], Gridded(Linear()))(r)
end


"""
This is just the difference between the pseudopotentials of different angularMomentum
"""
function eval_psp_semilocal_real(psp::PspTM, r::T, angularMomentum::Int, localAngularMom::Int) where {T <: Real}
    return eval_psp_local_real(psp,r,angularMomentum) - eval_psp_local_real(psp,r,localAngularMom)
end

"""
Creating the projector function described in eq. 19 in the [Troullier-Martins](https://doi.org/10.1103/PhysRevB.43.1993) paper.
Also in the paper, they additionally normalized by Ω.
More information on how ABINIT parses the file, see the [m_psp1](https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90) page Lines: 800 - 825
"""
function eval_psp_projector_fourier(psp::PspTM, q::T, qPrime::T, angularMomentum::Int) where {T <: Real} 
    function summations(r::T)
        nonLocalPotential(r) * pseudoWaveFunction(r) * besselj(angularMomentum,abs(q*r)) * r^2
    end
    radialGrid = [100*((x/2000)+0.1)^5-1e-8 for x in 0:2000]
    pseudoWaveFunction = interpolate((radialGrid,), psp.firstProjectorVals[anglularMomentum + 1], Gridded(Linear()))
    normalizer(r::Real) = integrate(pseudoWaveFunction * (eval_psp_semilocal_real * pseudoWaveFunction), 0, Inf)
    r = 2*pi * inv(q)
    integrate(summations, 0, Inf) * integrate(summations, 0, Inf) * 
        sum(ylm_real(angularMomentum,m,q) * conj(ylm_real(angularMomentum,m,qPrime)) for m in -angularMomentum:angularMomentum) / normalizer
end


"""
Creating the projector function described in eq. 10 in the [Troullier-Martins](https://doi.org/10.1103/PhysRevB.43.1993) paper.
More information on how ABINIT parses the file, see the [m_psp1](https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90) page Lines: 695 - 724
"""
function eval_psp_projector_real(psp::PspTm, r::T, angularMomentum::Int) where {T <: Real}
    radialGrid = [100*((x/2000)+0.1)^5-1e-8 for x in 0:2000]
    pseudoWaveFunction = interpolate((radialGrid,), psp.firstProjectorVals[anglularMomentum + 1], Gridded(Linear()))
    normalizer = integrate(pseudoWaveFunction * (eval_psp_semilocal_real * pseudoWaveFunction), 0, Inf)
    eval_psp_semilocal_real(r) = eval_psp_semilocal_real(psp,r,angularMomentum)
    waveFctPspSquared(r::T) = (pseudoWaveFunction(r) * eval_psp_semilocal_real(r))^2
    return waveFctPspSquared(r) / normalizer
end