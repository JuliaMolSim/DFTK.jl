using SpecialFunctions: besselj
using Interpolations: interpolate, Gridded, Linear
using Plots: plot, plot!
using QuadGK: quadgk, quadgk!, gauss
import PeriodicTable
using BenchmarkTools: @benchmark

struct PspTm <: NormConservingPsp
    Zion::              Int                           # Atomic number of atom
    valenceElectrons::  Int                           # Number of valence electrons
    lmax::              Int                           # Maximal angular momentum in the non-local part
    lloc::              Int                           # Angular momentum used for local part
    numGridPoints::     Int                           # Number of grid points used in the radial grid
    numProjectorFctns::     Vector{Int}               # Number of projector functions used for each angular momentum
    pspCoreRadius::         Vector{Float64}           # Pseudopotential core radius per angular momentum
    rms::                   Vector{Float64}           # Measure of pseudopotential quality reflecting the value of the penalty function in designing the potential per angular momentum
    energiesKB::            Vector{Vector{Float64}}   # ekb are Kleinman-Bylander energies for each projection function for each angular momentum
    epsatm::                Vector{Float64}           # The integral ∫^∞_0 4πr*(r*V(r)+valenceElectrons)dr
    rchrg::             Float64                       # the Core charge radius for additional core charge used to match the xc contribution
    fchrg::             Float64                       # The prefactor of the core charge expression
    totCoreChrg::       Float64                       # The total (integrated) core charge
    radialGrid::            Vector{Float64}           # grid used to generate the pseudopotential and the pseudowaves [100*((x/numGridPoints)+0.1)^5-1e-8 for x in 0:numGridPoints]
    pseudoPotentials::      Vector{Vector{Float64}}   # Radial pseudopotential values for each angular momentum
    projectorVals::         Vector{Vector{Vector{Float64}}}   # First and second Radial projection values for each angular momentum
    h::                     Array{Float64,2}          # h coefficients
    identifier::        String                        # String identifying the PSP
    description::       String                        # Descriptive string
end

function parse_tm_file(path; identifier="")
    #Information for where to find information about this pseudopotential file format can be found at https://docs.abinit.org/developers/psp1_info/, and all files were retrieved from https://www.abinit.org/sites/default/files/PrevAtomicData/psp-links/lda_tm.html. The example file used is hydrogen.
    file = readlines(path)
    description = file[1] #First line of the TM psp files is always the description of the file. This contains the type of pseudopotential, the element symbol, and the publish date.

    #Expecting the following format: '1.00000   1.00000    940714                zatom, zion, pspdat'
    Zatom, Zion = Int.(parse.(Float64,split(file[occursin.(r"zatom, zion", file)][1])[1:2]))
    @assert Zatom == PeriodicTable.elements[Zatom].number
    @assert Zion <= Zatom

    #Expecting the following format: 1    1    0    0      2001    .00000      pspcod,pspxc,lmax,lloc,mmax,r2well
    maximumAngularMomentum, localAngularMomentum, numGridPoints = parse.(Int,split(file[occursin.(r"lmax,lloc", file)][1])[3:5])
    @assert maximumAngularMomentum <= 3
    @assert localAngularMomentum <= maximumAngularMomentum
    @assert numGridPoints == 2001

    #Expecting the following format: 0   7.740  11.990    0   1.5855604        l,e99.0,e99.9,nproj,rcpsp
    #This format is the same for each line that contains r"l,e99.0"
    numProjectorFctns = []; pspCoreRadius = []
    foreach(file[occursin.(r"l,e99.0",file)]) do line
        nproj,rcpsp = parse.(Float64,split(line)[4:5])
        push!(numProjectorFctns,Int(nproj))
        push!(pspCoreRadius,rcpsp)
    end
    @assert length(numProjectorFctns) == maximumAngularMomentum + 1

    #Expecting the following format: .00000000    .0000000000    .0000000000    .00000000   rms,ekb1,ekb2,epsatm
    #This format is the same for each line that contains r"rms,ekb1"
    rms = []; energiesKB = []; epsatm = []
    foreach(file[occursin.(r"rms,ekb1",file)]) do line
        rmS,ekb1,ekb2,epsatM = parse.(Float64,split(line)[1:4])
        push!(rms,rmS)
        push!(energiesKB,[ekb1,ekb2])
        push!(epsatm,epsatM)
    end

    #Expecting the following format:.00000000000000     .00000000000000     .00000000000000   rchrg,fchrg,qchrg
    rchrg, fchrg, totCoreChrg = parse.(Float64,split(file[occursin.(r"rchrg,fchrg", file)][1])[1:3])

    radialGrid = [100*((x/Int(numGridPoints))+0.01)^5-1e-8 for x in 0:Int(numGridPoints-1)] #Creates the radial grid that the pseudopotentials and the pseudowaves are on.

    #Expecting the following format: -2.5567419069496231E+00  -2.5567419069496071E+00  -2.5567419069495880E+00
    # All of these files are expected to have 2001 pseudopotential and projector data points split into 3 columns making 667 rows of data.
    pseudoPotentials = map(findall(occursin.(r"for Troullier-Martins",file))) do idx
        parse.(Float64,string.(vcat(split.(file[idx+1:idx+667])...)))
    end
    @assert length(pseudoPotentials) == maximumAngularMomentum + 1
    @assert all(x -> length(x) == numGridPoints, pseudoPotentials)
    
    #Expecting the following format: 0.0000000000000000E+00   3.6287163307877053E-09   8.0185141095298299E-09
    firstProjectorVals = map(findall(occursin.(r"first projection function",file))) do idx
            parse.(Float64,string.(vcat(split.(file[idx+1:idx+667])...)))
    end
    secondProjectorVals = map(findall(occursin.(r"second projection function",file))) do idx
            parse.(Float64,string.(vcat(split.(file[idx+1:idx+667])...)))
    end
    projectorVals = [firstProjectorVals,secondProjectorVals]
    @assert length(projectorVals[1]) == maximumAngularMomentum + 1
    @assert all(x -> length(x) == numGridPoints, projectorVals[1])
    @assert length(projectorVals[2]) == count(x -> x > 1, numProjectorFctns)

    h = zeros(2,4)
    function hCoefficients(radialGrid,pseudoPotentials,projectorVals,projector, angularMomentum)
        f(r) = interpolate((radialGrid,),pseudoPotentials[angularMomentum+1], Gridded(Linear()))(r) * 
                (interpolate((radialGrid,),projectorVals[projector][angularMomentum+1],Gridded(Linear()))(r))^2
        return inv(first(quadgk(f,radialGrid[1],radialGrid[end]; order = 17, rtol = 1e-7)))
    end
    foreach(1:maximumAngularMomentum+1) do i
        h[1,i] = hCoefficients(radialGrid,pseudoPotentials,projectorVals, 1, i-1)
        isempty(projectorVals[2]) || (h[2,i] = hCoefficients(radialGrid, pseudoPotentials, projectorVals, 2, i-1))
    end
 
    #After the projector values, it's possible for some files to contain information about setting up the pseudopotential
    pspSetupDescription = "\nHere are some more details about setting up of the pseudopotential:\n"
    if isempty(projectorVals[2])
        if lastindex(file) > findlast(occursin.(r"first projection",file))+667
            description *= pspSetupDescription * prod(line -> line *" \n",file[findlast(occursin.(r"first projection",file)) + 668 : end])
        end
    else
        if lastindex(file) > findlast(occursin.(r"second projection",file))+667
            description *= pspSetupDescription * prod(line -> line *" \n",file[findlast(occursin.(r"second projection",file)) + 668 : end])
        end
    end
    
    PspTm(
        Zatom, Zion,
        maximumAngularMomentum, localAngularMomentum, numGridPoints,
        numProjectorFctns, pspCoreRadius,
        rms, energiesKB, epsatm,
        rchrg, fchrg, totCoreChrg,
        radialGrid,
        pseudoPotentials,
        projectorVals,
        h,
        identifier,
        description
    )
end

function eval_pseudoWaveFunction_real(psp::PspTm, projector::Int, angularMomentum::Int, r::T) where {T <: Real}
    @assert 0 <= projector <= 2
    projector == 2 && (return interpolate((psp.radialGrid,), psp.secondProjectorVals[angularMomentum + 1], Gridded(Linear()))(r))
    return interpolate((psp.radialGrid,), psp.projectorVals[projector][angularMomentum + 1], Gridded(Linear()))(r)
end

"""
This is just creating a function from the pseudopotential that came from the file. See [m_psp1](https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90) Line: 257 and 240
I have no real justification for using the mean pseudopotential instead of the pseudopotentials for an angular momentum. It's just more convienent (no angularMomentum input variable), and when taking the difference between a given pseudopotential of an angular momentum none of the potentials are zero.
"""
eval_psp_local_real(psp::PspTm, r::T) where {T <: Real} = interpolate((psp.radialGrid,), psp.pseudoPotentials[psp.lloc+1], Gridded(Linear()))(r)

#This was how ABINIT calculated their local fourier potential. See https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90 Lines: 434,435
@doc raw"""
Local potential in inverse space. Calculated with the Hankel transformation: 4π∫(\frac{sin(2π q r)}{2π q r})(r^2 v(r)+r Zv)dr.
"""
function eval_psp_local_fourier(psp::PspTm, q::T) where {T <: Real} #This is increadibly slow
    j0(r) = sin(2π * q * r)/(2π * q)
    f(r) = j0(r) * (r * eval_psp_local_real(psp,r) + psp.Zion)
    return 4π * quadgk(f, psp.radialGrid[1],psp.radialGrid[end]; order = 17, rtol = 1e-8)[1]
end

function approx(psp,q,N)
    W(r) = sin(2π * q * r)/(2π * q)
    g(r) = r * eval_psp_local_real(psp,r) + psp.Zion
    r,w = gauss(N,psp.radialGrid[1], psp.radialGrid[end])
    return dot(W.(r),w)
end

"""
This is just the difference between the pseudopotentials of different angularMomentum and the average of the pseudopotentials
"""
function eval_psp_semilocal_real(psp::PspTm, r::T, angularMomentum::Int) where {T <: Real}
    angularMomentum == psp.lloc && (return 0.0)
    psp_l = interpolate((psp.radialGrid,), psp.pseudoPotentials[angularMomentum + 1], Gridded(Linear()))
    return psp_l(r) - eval_psp_local_real(psp,r)
end

function eval_psp_energy_correction(psp::PspTm,r::T,angularMomentum::Int) where {T <: Real} #From https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90 Line: 658 in the psp1nl function
    f(r) = (eval_pseudoWaveFunction_real(psp,projector,angularMomentum,r) * eval_psp_semilocal_real(r)
    )^2
    return first(quadgk(f, psp.radialGrid[1], psp.radialGrid[end]; rtol = 1e-1))
end

#This is the normalizer that ABINIT uses for their nonlocal K-B potential
# https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90 Line: 696-713
function normalizer(psp, projector, angularMomentum)
    #The normalizer should have a square root, but it will be squared later on.
    angularMomentum == psp.lloc && (return 1.0)
    return quadgk(x -> (eval_pseudoWaveFunction_real(psp,projector,angularMomentum,x) * eval_psp_semilocal_real(psp,x,angularMomentum))^2, psp.radialGrid[1], psp.radialGrid[end]; rtol = 1e-1) |> first
end


"""
Creating the projector function described in eq. 10 in the [Troullier-Martins](https://doi.org/10.1103/PhysRevB.43.1993) paper.
    Eq. 10: 
More information on how ABINIT parses the file, see the [m_psp1](https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90) page Lines: 695 - 724
"""
function eval_psp_projector_real(psp::PspTm, projector::Int, angularMomentum::Int, r::T) where {T <: Real}
    psp.lloc == angularMomentum && (return 0.0)
    nonLocalPotential = (eval_pseudoWaveFunction_real(psp,projector,angularMomentum,r) * eval_psp_semilocal_real(psp,r,angularMomentum))^2 / normalizer(psp, projector, angularMomentum)
    return nonLocalPotential * psp.h[projector,angularMomentum+1]
end

"""
Creating the projector function described in eq. 19 in the [Troullier-Martins](https://doi.org/10.1103/PhysRevB.43.1993) paper.
Also in the paper, they additionally normalized by Ω.
More information on how ABINIT parses the file, see the [m_psp1](https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90) page Lines: 800 - 825
"""
function eval_psp_projector_fourier(psp::PspTm, projector::Int, angularMomentum::Int, q::T) where {T <: Real} 
    angularMomentum == psp.lloc && (return 0.0)
    x(r) = 2π * r * q 
    rDependencies(r) = r^2 * eval_psp_projector_real(psp, projector, angularMomentum, r)
    bess(r) = if angularMomentum == 0
        -besselj(1,x(r))
    else
        besselj(angularMomentum-1,x(r)) - (angularMomentum+1) * besselj(angularMomentum,x(r))/x(r)
    end
    return quadgk(r -> 2π * rDependencies(r) * bess(r), psp.radialGrid[1], psp.radialGrid[end]; rtol = 1e-1)|>first#, atol = 1e0)|> first
end