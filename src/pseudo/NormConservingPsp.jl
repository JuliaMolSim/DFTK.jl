# # Abstract interface for a norm-conserving pseudopotential. See PspHgh for an example.
# abstract type NormConservingPsp end

# # Subtypes must implement the following:

# #### Fields:

# #    lmax::Int                   # Maximal angular momentum in the non-local part
# #    h::Vector{Matrix{Float64}}  # Projector coupling coefficients per AM channel: h[l][i1,i2]
# #    identifier::String          # String identifying the PSP
# #    description::String         # Descriptive string

#### Methods:
# charge_ionic(psp::NormConservingPsp)
# has_valence_density(psp:NormConservingPsp)
# has_core_density(psp:NormConservingPsp)
# eval_psp_projector_real(psp::NormConservingPsp, i, l, r::Real)
# eval_psp_projector_fourier(psp::NormConservingPsp, i, l, q::Real)
# eval_psp_local_real(psp::NormConservingPsp, r::Real)
# eval_psp_local_fourier(psp::NormConservingPsp, q::Real)
# eval_psp_energy_correction(T::Type, psp::NormConservingPsp, n_electrons::Integer)

#### Optional methods:
# eval_psp_density_valence_real(psp::NormConservingPsp, r::Real)
# eval_psp_density_valence_fourier(psp::NormConservingPsp, q::Real)
# eval_psp_density_core_real(psp::NormConservingPsp, r::Real)
# eval_psp_density_core_fourier(psp::NormConservingPsp, q::Real)

"""
    eval_psp_projector_real(psp, i, l, r)

Evaluate the radial part of the `i`-th projector for angular momentum `l`
in real-space at the vector with modulus `r`.
"""
eval_psp_projector_real(psp::NormConservingPsp, i, l, r::AbstractVector) =
    eval_psp_projector_real(psp, i, l, norm(r))

@doc raw"""
    eval_psp_projector_fourier(psp, i, l, q)

Evaluate the radial part of the `i`-th projector for angular momentum `l`
at the reciprocal vector with modulus `q`:
```math
\begin{aligned}
p(q) &= ∫_{ℝ^3} p_{il}(r) e^{-iq·r} dr \\
     &= 4π ∫_{ℝ_+} r^2 p_{il}(r) j_l(qr) dr
\end{aligned}
```
"""
eval_psp_projector_fourier(psp::NormConservingPsp, q::AbstractVector) =
    eval_psp_projector_fourier(psp, norm(q))

"""
    eval_psp_local_real(psp, r)

Evaluate the local part of the pseudopotential in real space.
"""
eval_psp_local_real(psp::NormConservingPsp, r::AbstractVector) =
    eval_psp_local_real(psp, norm(r))

@doc raw"""
    eval_psp_local_fourier(psp, q)

Evaluate the local part of the pseudopotential in reciprocal space:
```math
\begin{aligned}
V_{\rm loc}(q) &= ∫_{ℝ^3} V_{\rm loc}(r) e^{-iqr} dr \\
               &= 4π ∫_{ℝ_+} V_{\rm loc}(r) \frac{\sin(qr)}{q} r dr
\end{aligned}
```
In practice, the local potential should be corrected using a Coulomb-like term ``C(r) = -Z/r``
to remove the long-range tail of ``V_{\rm loc}(r)`` from the integral:
```math
\begin{aligned}
V_{\rm loc}(q) &= ∫_{ℝ^3} (V_{\rm loc}(r) - C(r)) e^{-iq·r} dr + F[C(r)] \\
               &= 4π ∫_{ℝ_+} (V_{\rm loc}(r) + Z/r) \frac{\sin(qr)}{qr} r^2 dr - Z/q^2
\end{aligned}
```
"""
eval_psp_local_fourier(psp::NormConservingPsp, q::AbstractVector) =
    eval_psp_local_fourier(psp, norm(q))

@doc raw"""
    eval_psp_energy_correction([T=Float64,] psp, n_electrons)

Evaluate the energy correction to the Ewald electrostatic interaction energy of one unit
cell, which is required compared the Ewald expression for point-like nuclei. `n_electrons`
is the number of electrons per unit cell. This defines the uniform compensating background
charge, which is assumed here.

Notice: The returned result is the *energy per unit cell* and not the energy per volume.
To obtain the latter, the caller needs to divide by the unit cell volume.

The energy correction is defined as the limit of the Fourier-transform of the local
potential as ``q \to 0``, using the same correction as in the Fourier-transform of the local
potential:
```math
\lim_{q \to 0} 4π N_{\rm elec} ∫_{ℝ_+} (V(r) - C(r)) \frac{\sin(qr)}{qr} r^2 dr + F[C(r)]
 = 4π N_{\rm elec} ∫_{ℝ_+} (V(r) + Z/r) r^2 dr
 ```
"""
function eval_psp_energy_correction end
# by default, no correction, see PspHgh implementation and tests
# for details on what to implement
eval_psp_energy_correction(psp::NormConservingPsp, n_electrons) =
    eval_psp_energy_correction(Float64, psp, n_electrons)

"""
    eval_psp_density_valence_real(psp, r)

Evaluate the atomic valence charge density in real space.
"""
eval_psp_density_valence_real(psp::NormConservingPsp, r::AbstractVector) = 
    eval_psp_density_valence_real(psp, norm(r))

"""
    eval_psp_density_valence_fourier(psp, q)

Evaluate the atomic valence charge density in reciprocal space:
ρval(q) = ∫_{R^3} ρval(r) e^{-iqr} dr
        = 4π ∫_{R+} ρval(r) sin(qr)/qr r^2 dr
"""
eval_psp_density_valence_fourier(psp::NormConservingPsp, q::AbstractVector) = 
    eval_psp_density_valence_fourier(psp, norm(q))

"""
    eval_psp_density_core_real(psp, r)

Evaluate the atomic core charge density in real space.
"""
eval_psp_density_core_real(::NormConservingPsp, ::T) where {T <: Real} = zero(T)
eval_psp_density_core_real(psp::NormConservingPsp, r::AbstractVector) = 
    eval_psp_density_core_real(psp, norm(r))

"""
    eval_psp_density_core_fourier(psp, q)

Evaluate the atomic core charge density in reciprocal space:
ρval(q) = ∫_{R^3} ρcore(r) e^{-iqr} dr
        = 4π ∫_{R+} ρcore(r) sin(qr)/qr r^2 dr
"""
eval_psp_density_core_fourier(::NormConservingPsp, ::T) where {T <: Real} = zero(T)
eval_psp_density_core_fourier(psp::NormConservingPsp, q::AbstractVector) = 
    eval_psp_density_core_fourier(psp, norm(q))


#### Methods defined on a NormConservingPsp
import Base.Broadcast.broadcastable
Base.Broadcast.broadcastable(psp::NormConservingPsp) = Ref(psp)

function projector_indices(psp::NormConservingPsp)
    ((i, l, m) for l in 0:psp.lmax for i in 1:size(psp.h[l+1], 1) for m = -l:l)
end

"""
    count_n_proj_radial(psp, l)

Number of projector radial functions at angular momentum `l`.
"""
count_n_proj_radial(psp::NormConservingPsp, l::Integer) = size(psp.h[l + 1], 1)

"""
    count_n_proj_radial(psp)

Number of projector radial functions at all angular momenta up to `psp.lmax`.
"""
function count_n_proj_radial(psp::NormConservingPsp)
    sum(l -> count_n_proj_radial(psp, l), 0:psp.lmax; init=0)::Int
end

"""
    count_n_proj(psp, l)

Number of projector functions for angular momentum `l`, including angular parts from `-m:m`.
"""
count_n_proj(psp::NormConservingPsp, l::Integer) = count_n_proj_radial(psp, l) * (2l + 1)

"""
    count_n_proj(psp)

Number of projector functions for all angular momenta up to `psp.lmax`, including
angular parts from `-m:m`.
"""
function count_n_proj(psp::NormConservingPsp)
    sum(l -> count_n_proj(psp, l), 0:psp.lmax; init=0)::Int
end

"""
    count_n_proj(psps, psp_positions)

Number of projector functions for all angular momenta up to `psp.lmax` and for all
atoms in the system, including angular parts from `-m:m`.
"""
function PseudoPotentialIO.n_projector_angulars(psps, psp_positions)
    sum(PseudoPotentialIO.n_projector_angulars(psp) * length(positions)
        for (psp, positions) in zip(psps, psp_positions))
end
