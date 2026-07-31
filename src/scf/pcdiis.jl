@doc raw"""
First draft of Pcdiis implementation in DFTK

Pcdiis stands for Projected-Commutator-DIIS and is an attempt to make CDIIS feasible for large basis sets.
Instead of computing e = [H,ρ] in the full basis, this is only done in a subspace: ē = <ψ_ref|e|ψ_ref>
Instead of mixing full density or Fock matrices, 
the occupied states are mixed after gauge fixing (gf): ψ_gf = |ψ_occ><ψ_occ|ψ_ref_occ>

Solving for the coefficients is done just like for the original DIIS:

┌                  ┐   ┌   ┐   ┌   ┐
│  0   1   1  …  1 │   │ λ │   │ 1 │
│  1  B₁₁ B₁₂ … B₁ₙ│   │c₁ │   │ 0 │
│  1  B₂₁ B₂₂ … B₂ₙ│ · │c₂ │ = │ 0 │
│  ⋮   ⋮   ⋮  ⋱  ⋮ │   │ ⋮ │   │ ⋮ │
│  1  Bₙ₁ Bₙ₂ … Bₙₙ│   │cₙ │   │ 0 │
└                  ┘   └   ┘   └   ┘

where Bᵢⱼ = ⟨eᵢ|eⱼ⟩

I copied the layout of the anderson.jl file here

For now, there is no history management implemented except for a maximal depth.

[^HLY17]: Hu, Lin, Yang. Journal of chemical theory and computation **13.11**, 5458-5467 (2017) DOI [10.1021/acs.jctc.7b00892](https://doi.org/10.1021/acs.jctc.7b00892) 
"""

@kwdef struct PcdiisAcceleration
    errorhistory = Vector{Vector{Matrix{ComplexF64}}}() #error vectors
    statehistory = Vector{Vector{Matrix{ComplexF64}}}() #gauge fixed orbitals
    B            = Vector{Matrix{Float64}}()            #stores B_ij = Tr(e_ie_j*) for each kpoint

    depth::Int8                              = 10 #maximal depth of Pcdiis accelerator
    ψ_ref                                    = Vector{Matrix{ComplexF64}}() #reference states. 
    ψ_res                                    = Vector{Matrix{ComplexF64}}() #reference states. 
    switch                                   = false
    nb                                       = 0 

    #data collection for debugging:
    fock::Vector{Matrix{ComplexF64}}         = []
    history::Dict{String, Vector} = Dict(
        "err"           => Float64[],
        "energy"        => Float64[],
        "comu"          => Matrix{ComplexF64}[],
        "fock"          => Matrix{ComplexF64}[],
        "over1"         => Matrix{ComplexF64}[],
        "over2"         => Matrix{ComplexF64}[],
        "over3"         => Matrix{ComplexF64}[],
        "rho"           => Matrix{ComplexF64}[],
        "sys"           => Matrix{Float64}[],
    )
end

function Base.deleteat!(pcdiis::PcdiisAcceleration, idx)
   deleteat!(pcdiis.errorhistory, idx)
   deleteat!(pcdiis.statehistory, idx)
   pcdiis
end

function Base.popfirst!(pcdiis::PcdiisAcceleration)
   popfirst!(pcdiis.errorhistory)
   popfirst!(pcdiis.statehistory)
   pcdiis
end

function Base.push!(pcdiis::PcdiisAcceleration, cₙ, ψ_gfₙ)
    push!(pcdiis.errorhistory,  cₙ)
    push!(pcdiis.statehistory, ψ_gfₙ)

    length(pcdiis.errorhistory) > pcdiis.depth && popfirst!(pcdiis)
    @debug "Pcdiis depth: $(length(pcdiis.errorhistory))"
    @assert length(pcdiis.errorhistory) <= pcdiis.depth
    @assert length(pcdiis.errorhistory) == length(pcdiis.statehistory)
    pcdiis
end

@timing "Pcdiis acceleration" function (pcdiis::PcdiisAcceleration)(fxₙ, infoₙ,finfoₙ)
	if pcdiis.depth <= 1 || isnothing(infoₙ.ψ)
		return fxₙ, finfoₙ
	end

	ψ = finfoₙ.ψ
	ψ_old = infoₙ.ψ
	occ = finfoₙ.occupation
	H = finfoₙ.ham
    C=0
    hamil=0
    ρ=0

    isempty(pcdiis.ψ_ref) && append!(pcdiis.ψ_ref, [deepcopy(ψk[:, begin:max(pcdiis.nb, size(ψ_old,2))]) for ψk in ψ_old])
    #isempty(pcdiis.ψ_res) && append!(pcdiis.ψ_res, [deepcopy(ψk[:, begin:max(pcdiis.nb, size(ψ_old,2))]) for ψk in ψ_old])

	#creating mask to obtain occupied orbitals only
	#will fail if non-integer occupations are provided
	mask = occ[1] .> 0
	old_length = length(mask)
	mask_ref = deepcopy(mask)
	resize!(mask_ref, size(pcdiis.ψ_ref[1],2))
	mask_ref[old_length+1:end] .= false

	k_errors::Vector = []
	k_states::Vector = []
	for ik in 1:length(ψ)
		#compute gauge-fixed states
		push!(k_states, ψ[ik][:,mask] * (ψ[ik][:,mask]' * pcdiis.ψ_ref[ik][:,mask_ref]))

		#Compute and save error vector
		ψ_ref_H_ψ_old = pcdiis.ψ_ref[ik]' * (H.blocks[ik] * ψ_old[ik][:,mask])
		ψ_old_ψ_ref   = ψ_old[ik][:,mask]' * pcdiis.ψ_ref[ik]
		C = ψ_ref_H_ψ_old * ψ_old_ψ_ref - ψ_old_ψ_ref' * ψ_ref_H_ψ_old'

        ρ = ψ_old_ψ_ref' * ψ_old_ψ_ref
        hamil = pcdiis.ψ_ref[ik]' * (H.blocks[ik] * pcdiis.ψ_ref[ik])

		if isempty(pcdiis.errorhistory)
			push!(pcdiis.B, ones(Float64,1,1) .* norm(C)^2)
		end
		push!(k_errors, C)
	end

    #push!(pcdiis.fock, pcdiis.ψ_ref[1]' * (finfoₙ.ham.blocks[1] * pcdiis.ψ_ref[1]))

	if isempty(pcdiis.errorhistory) 
		push!(pcdiis, k_errors, k_states)
        push!(pcdiis.history["err"   ], pcdiis.B[1][end, end])
        push!(pcdiis.history["energy"], Float64(finfoₙ.energies.total))
        push!(pcdiis.history["comu"  ], C)
        push!(pcdiis.history["fock"  ], C)
        push!(pcdiis.history["rho"   ], ρ)
        push!(pcdiis.history["sys"   ], pcdiis.B[1])

		return fxₙ, finfoₙ
	else
		push!(pcdiis, k_errors, k_states)
	end

	#building new matrices B_ij, only computing the latest column/line and reusing the rest
	lnew = length(pcdiis.errorhistory)
	for ik in 1:length(ψ)

		b = zeros(Float64,lnew,lnew)
		b_iend = [real(tr(pcdiis.errorhistory[ii][ik]' * pcdiis.errorhistory[end][ik])) 
			  for ii in 1:length(pcdiis.errorhistory)]

		lnew = length(b_iend)

		b[end,1:end] = b_iend
		b[1:end,end] = b_iend
		b[1:end-1,1:end-1] = pcdiis.B[ik][end-lnew+2:end,end-lnew+2:end]
		
		pcdiis.B[ik] = b
	end

	#for now, only one kpoint
	B = pcdiis.B[1]

	#solving for coefficients
	lb = size(B,1)+1
	sys = zeros(Float64,lb,lb)
	sys[2:end,2:end] = B
	sys[2:end,1] .= 1.0
	sys[1,2:end] .= 1.0
	rhs = zeros(lb)
	rhs[1] = 1.0
	
	coeffs = sys \ rhs
	cs = coeffs[2:end]

		for ik in 1:length(ψ)
			#mixing of pw coefficients of the gauge fixed states ψ_gf
			mixstate = zeros(ComplexF64, size(ψ[ik][:,mask])...)
			for ii in 1:length(cs)
				axpy!(cs[ii],pcdiis.statehistory[ii][ik],mixstate)
			end

			#not sure how to deal with this step exactly yet...
			#q = Matrix(qr(mixstate).Q)
			#q_ψ = q' * ψ[ik][:,mask]
			#u,s,v = svd(q_ψ)
			#rotation = u * v'
			#result = q * rotation

			#ψ[ik][:,mask] = result
			ψ[ik][:,mask] = mixstate
			ψ[ik] = Matrix(qr(ψ[ik]).Q)
		end

    push!(pcdiis.history["err"   ], pcdiis.B[1][end, end])
    push!(pcdiis.history["energy"], Float64(finfoₙ.energies.total))
    push!(pcdiis.history["comu"  ], C)
    push!(pcdiis.history["fock"  ], C)
    push!(pcdiis.history["rho"   ], ρ)
    push!(pcdiis.history["sys"   ], pcdiis.B[1])

	#density needs to be updated after mixing states
	return compute_density(H.basis, ψ, occ), finfoₙ
end
