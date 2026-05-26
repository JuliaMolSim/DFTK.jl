@doc raw"""
First draft of pcdiis implementation in DFTK
"""

@kwdef struct PcdiisAcceleration
    errorhistory::Vector = [] #error vectors
    statehistory::Vector = [] #gauge fixed orbitals

    depth::Int           = 10
    nelec::Int           = 0
    ψ_ref                = nothing
    B::Vector            = []
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

#take care of the arguments of that function later
function Base.push!(pcdiis::PcdiisAcceleration, cₙ, ψ_projₙ)
    push!(pcdiis.errorhistory,  cₙ)
    push!(pcdiis.statehistory, ψ_projₙ)

    length(pcdiis.errorhistory) > pcdiis.depth && popfirst!(pcdiis)
    @debug "Pcdiis depth: $(length(pcdiis.errorhistory))"
    @assert length(pcdiis.errorhistory) <= pcdiis.depth
    @assert length(pcdiis.errorhistory) == length(pcdiis.statehistory)
    pcdiis
end

@timing "Pcdiis acceleration" function (pcdiis::PcdiisAcceleration)(fxₙ, infoₙ,finfoₙ)
	if pcdiis.depth <= 1 || isnothing(pcdiis.ψ_ref) || isnothing(infoₙ.ψ)
		return fxₙ, finfoₙ
	end

	ψ = finfoₙ.ψ
	print(size(ψ[1]))
	print(pcdiis.nelec)
	print("\n")
	ψ_old = infoₙ.ψ
	occ = finfoₙ.occupation
	eig = finfoₙ.eigenvalues
	H = finfoₙ.ham

	#bodgy, need to do this for each kpoint separately
	mask = occ[1] .> 0
	print(mask)
	print("\n")
	mask[1:pcdiis.nelec] .= true
	print(mask)
	print("\n")
	mask[pcdiis.nelec+1:end] .= false
	print(mask)
	print("\n")
	mask_ref = deepcopy(mask)
	resize!(mask_ref, size(pcdiis.ψ_ref[1],2))
	mask_ref[pcdiis.nelec+1:end] .= false
	print(mask_ref)
	print("\n")

	k_errors::Vector = []
	k_states::Vector = []
	for ik in 1:length(ψ)
		#compute gauge-fixed states
		push!(k_states, ψ[ik][:,mask] * (ψ[ik][:,mask]' * pcdiis.ψ_ref[ik][:,mask_ref]))

		#Compute and save error vector
		ψ_ref_H_ψ_old = pcdiis.ψ_ref[ik]' * (H.blocks[ik] * ψ_old[ik][:,mask])
		ψ_old_ψ_ref   = ψ_old[ik][:,mask]' * pcdiis.ψ_ref[ik]
		C = ψ_ref_H_ψ_old * ψ_old_ψ_ref - ψ_old_ψ_ref' * ψ_ref_H_ψ_old'
		if isempty(pcdiis.errorhistory)
			push!(pcdiis.B, ones(Float64,1,1) .* real(tr(C' * C)))
		end
		push!(k_errors, C)
	end

	if isempty(pcdiis.errorhistory) 
		push!(pcdiis, k_errors, k_states)
		return fxₙ, finfoₙ
	else
		push!(pcdiis, k_errors, k_states)
	end

	lnew = length(pcdiis.errorhistory)
	for ik in 1:length(ψ)

		b = zeros(Float64,lnew,lnew)
		b_iend = [real(tr(pcdiis.errorhistory[ii][ik]' * pcdiis.errorhistory[end][ik])) for ii in 1:length(pcdiis.errorhistory)]

		lnew = length(b_iend)

		b[end,1:end] = b_iend
		b[1:end,end] = b_iend
		b[1:end-1,1:end-1] = pcdiis.B[ik][end-lnew+2:end,end-lnew+2:end]
		
		pcdiis.B[ik] = deepcopy(b)
	end

	B = pcdiis.B[1]

	lb = size(B,1)+1
	sys = zeros(Float64,lb,lb)
	sys[2:end,2:end] = B
	sys[2:end,1] .= 1.0
	sys[1,2:end] .= 1.0
	rhs = zeros(lb)
	rhs[1] = 1
	
	coeffs = sys \ rhs
	cs = coeffs[2:end]

	for ik in 1:length(ψ)
		#the actual mixing of pw coefficients
		mixstate = zeros(ComplexF64, size(ψ[ik][:,mask])...)
		print(size(mixstate))
		print("\n")
		print(size(pcdiis.statehistory[1][1]))
		print("\n")
		for ii in 1:length(cs)
			axpy!(cs[ii],pcdiis.statehistory[ii][ik],mixstate)
		end

		#orthonormalization
		ψ[ik][:,mask] = mixstate
		ψ[ik] = Matrix(qr(ψ[ik]).Q)
	end

	return compute_density(H.basis, ψ, occ), finfoₙ
end
