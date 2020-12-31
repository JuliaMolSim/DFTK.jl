struct PspTM
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

# function PspTM(Zion, rloc, cloc, rp, h::Vector{Matrix{T}};
#                 identifier="", description="") where T
#     @assert length(rp) == length(h) "Length of rp and h do not agree"
#     lmax = length(h) - 1

#     @assert length(cloc) <= 4 "length(cloc) > 4 not supported."
#     if length(cloc) < 4
#         n_extra = 4 - length(cloc)
#         cloc = [cloc; zeros(n_extra)]
#     end

#     PspHgh(Zion, rloc, cloc, lmax, rp, h, identifier, description)
# end

function parse_TM_file(path; identifier="")
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
            
    function fct(vect)
        @show num = length(vect[1]) - 1
        r = map(x -> 100*(x/num + 0.01)^5 - 1e-8, eachindex(vect[1]))
        plt = plot()
        foreach(p -> display(plot!(plt,r,p)), vect)
    end
    # fct(psp)
    # fct(projfct)
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