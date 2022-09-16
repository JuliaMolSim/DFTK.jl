using Interpolations: LinearInterpolation

struct PspUpf{T} <: NormConservingPsp
    Zion::Int             # Ionic charge (Z - valence electrons)
    # ? More
    lmax::Int # Maximal angular momentum in the non-local part
    rgrid::Vector{T} # radial grid start from 0.0, mesh point
    lpot::Vector{T} # local potential in the radial grid
    projs::Vector{Vector{Vector{T}}} # BKB projector per AM per projector
    # ekb -> h??
    # ekb::Vector{Vector{T}} # KB energies per AM channel per projector (or store as matrix Dij?)
    # More? 
    h::Vector{Matrix{T}}  # Projector coupling coefficients per AM channel: h[l][i1,i2]
    identifier::String    # String identifying the PSP
    description::String   # Descriptive string
end
charge_ionic(psp::PspHgh) = psp.Zion

function parse_upf_file(path; identifier=path)
    # use UPF.jl by austin
    # h <- dij
end

"""
Read and interpolate value of projector in real space at r.
i-th projector of AM l. stored in file. Read from UPF file and interpolate.
"""
function eval_psp_projector_real(psp::PspUpf, i, l, r::T) where {T <: Real}
    interp = LinearInterpolation(psp.rgrid, psp.projs[l][i])
    interp(r)
end

function eval_psp_projector_fourier(psp::PspUpf, i, l, q::T) where {T <: Real}

end

function eval_psp_local_real(psp::PspUpf, r::T) where {T <: Real}
    interp = LinearInterpolation(psp.rgrid, psp.lpot)
    interp(r)
end

function eval_psp_local_fourier(psp::PspUpf, q::T) where {T <: Real}

end