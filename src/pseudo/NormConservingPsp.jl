abstract type NormConservingPsp end
# Subtypes must implement the following fields :
#    lmax::Int                   # Maximal angular momentum in the non-local part
#    h::Vector{Matrix{Float64}}  # Projector coupling coefficients per AM channel
#    Zion::Int                   # Ionic charge (Z - valence electrons)
#    identifier::String          # String identifying the PSP
#    description::String         # Descriptive string

# and methods:

# i-th projector for the angular momentum l p_{il}, in real space
eval_psp_projection_radial_real(psp::NormConservingPsp, i, l, r::T) where {T <: Real} =
    error("Not implemented")
eval_psp_projection_radial_real(psp::NormConservingPsp, i, l, r::AbstractVector) = eval_psp_projection_radial_real(psp, i, l, norm(r))
# i-th projector for the angular momentum l, in Fourier space
# p(q) = ∫_R^3 p_{il}(r) e^{-iqr} dr
#      = 4π ∫_{R+} r^2 p_{il}(r) j_l(q r) dr
eval_psp_projection_radial_fourier(psp::NormConservingPsp, i, l, q::T) where {T <: Real} =
    error("Not implemented")
eval_psp_projection_radial_fourier(psp::NormConservingPsp, q::AbstractVector) = eval_psp_projection_radial_fourier(psp, norm(q))

# local potential, in real space
eval_psp_local_real(psp::NormConservingPsp, r::T) where {T <: Real} =
    error("Not implemented")
eval_psp_local_real(psp::NormConservingPsp, r::AbstractVector) = eval_psp_local_real(psp, norm(r))
# local potential, in reciprocal space:
# V(q) = ∫_R^3 Vloc(r) e^{-iqr} dr
#      = 4π ∫_{R+} sin(qr)/q r e^{-iqr} dr
eval_psp_local_fourier(psp::NormConservingPsp, q::T) where {T <: Real} = 
    error("Not implemented")
eval_psp_local_fourier(psp::NormConservingPsp, q::AbstractVector) = eval_psp_local_fourier(psp, norm(q))


# optionally:
# energy correction: by default, no correction
eval_psp_energy_correction(T, psp::NormConservingPsp, n_electrons) = zero(T)

import Base.Broadcast.broadcastable
Base.Broadcast.broadcastable(psp::NormConservingPsp) = Ref(psp)

function projector_indices(psp::NormConservingPsp)
    ((i, l, m) for l in 0:psp.lmax for i in 1:size(psp.h[l+1], 1)
     for m = -l:l)
end

eval_psp_energy_correction(psp::NormConservingPsp, n_electrons) =
    eval_psp_energy_correction(Float64, psp, n_electrons)

# Number of projection vectors per atom
function count_n_proj(psp::NormConservingPsp)
    psp.lmax < 0 ? 0 : sum(size(psp.h[l + 1], 1) * (2l + 1) for l in 0:psp.lmax)::Int
end
function count_n_proj(atoms)
    sum(count_n_proj(psp)*length(positions) for (psp, positions) in atoms)::Int
end
