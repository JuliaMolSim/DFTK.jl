# Abstract interface for a norm-conserving pseudopotential. See PspHgh for an example.
abstract type NormConservingPsp end

# Subtypes must implement the following:

#### Fields:

#    lmax::Int                   # Maximal angular momentum in the non-local part
#    h::Vector{Matrix{Float64}}  # Projector coupling coefficients per AM channel: h[l][i1,i2]
#    identifier::String          # String identifying the PSP
#    description::String         # Descriptive string

#### Methods:
# charge_ionic(psp::NormConservingPsp)
# eval_psp_projector_real(psp::NormConservingPsp, i, l, r::Real)
# eval_psp_projector_fourier(psp::NormConservingPsp, i, l, q::Real)
# eval_psp_local_real(psp::NormConservingPsp, r::Real)
# eval_psp_local_fourier(psp::NormConservingPsp, q::Real)


"""
    eval_psp_projector_real(psp, i, l, r)

Evaluate the radial part of the `i`-th projector for angular momentum `l`
in real-space at the vector with modulus `r`.
"""
eval_psp_projector_real(psp::NormConservingPsp, i, l, r::AbstractVector) =
    eval_psp_projector_real(psp, i, l, norm(r))

"""
    eval_psp_projector_fourier(psp, i, l, q)

Evaluate the radial part of the `i`-th projector for angular momentum `l`
at the reciprocal vector with modulus `q`:
p(q) = ∫_R^3 p_{il}(r) e^{-iqr} dr
     = 4π ∫_{R+} r^2 p_{il}(r) j_l(q r) dr
"""
eval_psp_projector_fourier(psp::NormConservingPsp, q::AbstractVector) =
    eval_psp_projector_fourier(psp, norm(q))

"""
    eval_psp_local_real(psp, r)

Evaluate the local part of the pseudopotential in real space.
"""
eval_psp_local_real(psp::NormConservingPsp, r::AbstractVector) =
    eval_psp_local_real(psp, norm(r))

"""
    eval_psp_local_fourier(psp, q)

Evaluate the local part of the pseudopotential in reciprocal space:
V(q) = ∫_R^3 Vloc(r) e^{-iqr} dr
     = 4π ∫_{R+} Vloc(r) sin(qr)/q r dr
"""
eval_psp_local_fourier(psp::NormConservingPsp, q::AbstractVector) =
    eval_psp_local_fourier(psp, norm(q))

"""
    eval_psp_energy_correction([T=Float64,] psp, n_electrons)

Evaluate the energy correction to the Ewald electrostatic interaction energy of one unit
cell, which is required compared the Ewald expression for point-like nuclei. `n_electrons`
is the number of electrons per unit cell. This defines the uniform compensating background
charge, which is assumed here.

Notice: The returned result is the *energy per unit cell* and not the energy per volume.
To obtain the latter, the caller needs to divide by the unit cell volume.
"""
eval_psp_energy_correction(T, psp::NormConservingPsp, n_electrons) = zero(T)
# by default, no correction, see PspHgh implementation and tests
# for details on what to implement
eval_psp_energy_correction(psp::NormConservingPsp, n_electrons) =
    eval_psp_energy_correction(Float64, psp, n_electrons)


#### Methods defined on a NormConservingPsp
import Base.Broadcast.broadcastable
Base.Broadcast.broadcastable(psp::NormConservingPsp) = Ref(psp)

function projector_indices(psp::NormConservingPsp)
    ((i, l, m) for l in 0:psp.lmax for i in 1:size(psp.h[l+1], 1) for m = -l:l)
end

# Number of radial projection functions per atom
function count_n_proj_radial(psp::NormConservingPsp, l::Integer)
     psp.lmax < l < 0 ? 0 : size(psp.h[l + 1], 1)
end
function count_n_proj_radial(psp::NormConservingPsp)
    sum(count_n_proj(psp, l) for l in 0:psp.lmax)::Int
end
# Number of projection vectors per atom
function count_n_proj(psp::NormConservingPsp, l::Integer)
    count_n_proj_radial(psp, l) * (2l + 1)
end
function count_n_proj(psp::NormConservingPsp)
    sum(count_n_proj(psp, l) for l in 0:psp.lmax)::Int
end
function count_n_proj(psps, psp_positions)
    sum(count_n_proj(psp) * length(positions)
        for (psp, positions) in zip(psps, psp_positions))
end
