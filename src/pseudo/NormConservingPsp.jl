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
# has_valence_density(psp:NormConservingPsp)
# has_core_density(psp:NormConservingPsp)
# eval_psp_projector_real(psp::NormConservingPsp, i, l, r::Real)
# eval_psp_projector_fourier(psp::NormConservingPsp, i, l, p::Real)
# eval_psp_local_real(psp::NormConservingPsp, r::Real)
# eval_psp_local_fourier(psp::NormConservingPsp, p::Real)
# eval_psp_energy_correction(T::Type, psp::NormConservingPsp, n_electrons::Integer)

#### Optional methods:
# eval_psp_density_valence_real(psp::NormConservingPsp, r::Real)
# eval_psp_density_valence_fourier(psp::NormConservingPsp, p::Real)
# eval_psp_density_core_real(psp::NormConservingPsp, r::Real)
# eval_psp_density_core_fourier(psp::NormConservingPsp, p::Real)

"""
    eval_psp_projector_real(psp, i, l, r)

Evaluate the radial part of the `i`-th projector for angular momentum `l`
in real-space at the vector with modulus `r`.
"""
eval_psp_projector_real(psp::NormConservingPsp, i, l, r::AbstractVector) =
    eval_psp_projector_real(psp, i, l, norm(r))

@doc raw"""
    eval_psp_projector_fourier(psp, i, l, p)

Evaluate the radial part of the `i`-th projector for angular momentum `l`
at the reciprocal vector with modulus `p`:
```math
\begin{aligned}
{\rm proj}(p) &= ∫_{ℝ^3} {\rm proj}_{il}(r) e^{-ip·r} dr \\
              &= 4π ∫_{ℝ_+} r^2 {\rm proj}_{il}(r) j_l(p·r) dr.
\end{aligned}
```
"""
eval_psp_projector_fourier(psp::NormConservingPsp, p::AbstractVector) =
    eval_psp_projector_fourier(psp, norm(p))

"""
    eval_psp_local_real(psp, r)

Evaluate the local part of the pseudopotential in real space.
"""
eval_psp_local_real(psp::NormConservingPsp, r::AbstractVector) =
    eval_psp_local_real(psp, norm(r))

@doc raw"""
    eval_psp_local_fourier(psp, p)

Evaluate the local part of the pseudopotential in reciprocal space:
```math
\begin{aligned}
V_{\rm loc}(p) &= ∫_{ℝ^3} V_{\rm loc}(r) e^{-ip·r} dr \\
               &= 4π ∫_{ℝ_+} V_{\rm loc}(r) \frac{\sin(p·r)}{p} r dr
\end{aligned}
```
In practice, the local potential should be corrected using a Coulomb-like term ``C(r) = -Z/r``
to remove the long-range tail of ``V_{\rm loc}(r)`` from the integral:
```math
\begin{aligned}
V_{\rm loc}(p) &= ∫_{ℝ^3} (V_{\rm loc}(r) - C(r)) e^{-ip·r} dr + F[C(r)] \\
               &= 4π ∫_{ℝ_+} (V_{\rm loc}(r) + Z/r) \frac{\sin(p·r)}{p·r} r^2 dr - Z/p^2.
\end{aligned}
```
"""
eval_psp_local_fourier(psp::NormConservingPsp, p::AbstractVector) =
    eval_psp_local_fourier(psp, norm(p))

@doc raw"""
    eval_psp_energy_correction([T=Float64,] psp, n_electrons)

Evaluate the energy correction to the Ewald electrostatic interaction energy of one unit
cell, which is required compared the Ewald expression for point-like nuclei. `n_electrons`
is the number of electrons per unit cell. This defines the uniform compensating background
charge, which is assumed here.

Notice: The returned result is the *energy per unit cell* and not the energy per volume.
To obtain the latter, the caller needs to divide by the unit cell volume.

The energy correction is defined as the limit of the Fourier-transform of the local
potential as ``p \to 0``, using the same correction as in the Fourier-transform of the local
potential:
```math
\lim_{p \to 0} 4π N_{\rm elec} ∫_{ℝ_+} (V(r) - C(r)) \frac{\sin(p·r)}{p·r} r^2 dr + F[C(r)]
 = 4π N_{\rm elec} ∫_{ℝ_+} (V(r) + Z/r) r^2 dr.
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

@doc raw"""
    eval_psp_density_valence_fourier(psp, p)

Evaluate the atomic valence charge density in reciprocal space:
```math
\begin{aligned}
ρ_{\rm val}(p) &= ∫_{ℝ^3} ρ_{\rm val}(r) e^{-ip·r} dr \\
               &= 4π ∫_{ℝ_+} ρ_{\rm val}(r) \frac{\sin(p·r)}{ρ·r} r^2 dr.
\end{aligned}
```
"""
eval_psp_density_valence_fourier(psp::NormConservingPsp, p::AbstractVector) = 
    eval_psp_density_valence_fourier(psp, norm(p))

"""
    eval_psp_density_core_real(psp, r)

Evaluate the atomic core charge density in real space.
"""
eval_psp_density_core_real(::NormConservingPsp, ::T) where {T <: Real} = zero(T)
eval_psp_density_core_real(psp::NormConservingPsp, r::AbstractVector) = 
    eval_psp_density_core_real(psp, norm(r))

@doc raw"""
    eval_psp_density_core_fourier(psp, p)

Evaluate the atomic core charge density in reciprocal space:
```math
\begin{aligned}
ρ_{\rm core}(p) &= ∫_{ℝ^3} ρ_{\rm core}(r) e^{-ip·r} dr \\
               &= 4π ∫_{ℝ_+} ρ_{\rm core}(r) \frac{\sin(p·r)}{ρ·r} r^2 dr.
\end{aligned}
```
"""
eval_psp_density_core_fourier(::NormConservingPsp, ::T) where {T <: Real} = zero(T)
eval_psp_density_core_fourier(psp::NormConservingPsp, p::AbstractVector) = 
    eval_psp_density_core_fourier(psp, norm(p))


#### Methods defined on a NormConservingPsp
import Base.Broadcast.broadcastable
Base.Broadcast.broadcastable(psp::NormConservingPsp) = Ref(psp)

function projector_indices(psp::NormConservingPsp)
    ((i, l, m) for l = 0:psp.lmax for i = 1:size(psp.h[l+1], 1) for m = -l:l)
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
function count_n_proj(psps, psp_positions)
    sum(count_n_proj(psp) * length(positions)
        for (psp, positions) in zip(psps, psp_positions))
end
