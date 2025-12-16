# Linear combination of two pseudos
#    Note that this data structure is deliberately not yet exported for now
#    as we may want to find a different solution for representing virtual crystal elements
#    in the future; users should use virtual_crystal_element to implicitly construct
#    these objects.
#
struct PspLinComb{T, P <: NormConservingPsp} <: NormConservingPsp
    lmax::Int
    h::Vector{Matrix{T}}
    hidx_to_psp_proj::Vector{Vector{Tuple{Int,Int}}}  # map per angular momentum and projector index of this object to the tuple (psp index, projector index)
    coefficients::Vector{T}
    pseudos::Vector{P}
    identifier::String          # String identifying the PSP
    description::String         # Descriptive string
end

function PspLinComb(coefficients::Vector{T}, pseudos::Vector{<:NormConservingPsp};
                    description="psp linear combination") where {T}
    @assert length(coefficients) == length(pseudos)
    @assert sum(coefficients) ≈ one(eltype(coefficients))
    @assert !any(p -> p isa PspLinComb, pseudos)

    # These assumptions we could lift, but we make them to make our lifes easier for now
    @assert allequal(charge_ionic,        pseudos)
    @assert allequal(has_valence_density, pseudos)
    @assert allequal(has_core_density,    pseudos)
    @assert allequal(p -> p.lmax,         pseudos)

    TT   = promote_type(T, eltype(eltype(first(pseudos).h)))
    lmax = first(pseudos).lmax
    h = Matrix{TT}[]
    hidx_to_psp_proj = Vector{Tuple{Int,Int}}[]  # (ipsp, iproj)
    for l in 0:lmax
        proj_l_total = sum(p -> count_n_proj_radial(p, l), pseudos)
        hl = zeros(T, proj_l_total, proj_l_total)
        splits = Tuple{Int,Int}[]  # (ipsp, iproj)

        for (ipsp, p) in enumerate(pseudos)
            iproj_offset = isempty(splits) ? 0 : last(splits)[2]
            rnge = iproj_offset .+ (1:count_n_proj_radial(p, l))
            hl[rnge, rnge] = p.h[l+1] * coefficients[ipsp]
            append!(splits, [(ipsp, iproj) for iproj in 1:count_n_proj_radial(p, l)])
        end
        push!(h, hl)
        push!(hidx_to_psp_proj, splits)
    end
    @assert length(h) == length(hidx_to_psp_proj) == lmax+1

    identifier = ""
    PspLinComb(lmax, h, hidx_to_psp_proj, coefficients, pseudos, identifier, description)
end

charge_ionic(psp::PspLinComb)        = charge_ionic(first(psp.pseudos))
has_valence_density(psp::PspLinComb) = all(has_valence_density, psp.pseudos)
has_core_density(psp::PspLinComb)    = any(has_core_density,    psp.pseudos)

function eval_psp_projector_real(psp::PspLinComb, i, l, r::Real)
    (ipsp, iproj) = psp.hidx_to_psp_proj[l+1][i]
    eval_psp_projector_real(psp.pseudos[ipsp], iproj, l, r)
end
function eval_psp_projector_fourier(psp, i, l, p::Real)
    (ipsp, iproj) = psp.hidx_to_psp_proj[l+1][i]
    eval_psp_projector_fourier(psp.pseudos[ipsp], iproj, l, p)
end

function eval_psp_energy_correction(T, psp::PspLinComb)
    sum(c * eval_psp_energy_correction(T, p) for (c, p) in zip(psp.coefficients, psp.pseudos))
end

macro make_psplincomb_call(fn)
    quote
        function $fn(psp::PspLinComb, arg::Real)
            sum(c * $fn(pp, arg) for (c, pp) in zip(psp.coefficients, psp.pseudos))
        end
    end
end
@make_psplincomb_call DFTK.eval_psp_local_real
@make_psplincomb_call DFTK.eval_psp_local_fourier
@make_psplincomb_call DFTK.eval_psp_density_valence_real
@make_psplincomb_call DFTK.eval_psp_density_valence_fourier
@make_psplincomb_call DFTK.eval_psp_density_core_real
@make_psplincomb_call DFTK.eval_psp_density_core_fourier
