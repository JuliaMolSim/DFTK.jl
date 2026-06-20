# Abstract interface for a norm-conserving pseudopotential. See PspHgh for an example.
abstract type NormConservingPsp end

# Subtypes must implement the following:

#### Fields:

#    lmax::Int                   # Maximal angular momentum in the non-local part
#    h::Vector{Matrix{Float64}}  # Projector coupling coefficients per AM channel: h[l][i1,i2]
#    identifier::String          # String identifying the PSP
#    description::String         # Descriptive string

#### Required methods:
# charge_ionic(psp)
# has_valence_density(psp)
# has_core_density(psp)
# has_core_kinetic_energy_density(psp)
# eval_psp_projector_real(psp, i, l, r::Real)
# eval_psp_projector_fourier(psp, i, l, p::Real)
# eval_psp_local_real(psp, r::Real)
# eval_psp_local_fourier(psp, p::Real)
# eval_psp_energy_correction(T::Type, psp)

#### Optional methods:
# eval_psp_valence_density_real(psp, r::Real)
# eval_psp_valence_density_fourier(psp, p::Real)
# eval_psp_core_density_real(psp, r::Real)
# eval_psp_core_density_fourier(psp, p::Real)
# eval_psp_core_kinetic_energy_density_real(psp, r::Real)
# eval_psp_core_kinetic_energy_density_fourier(psp, p::Real)
# eval_psp_pswfc_real(psp, i::Int, l::Int, p::Real)
# eval_psp_pswfc_fourier(psp, i::Int, l::Int, p::Real)
# count_n_pswfc_radial(psp, l::Integer)
# pswfc_label(psp, i::Integer, l::Integer)

#
# Function stubs for required methods
#

"""
    eval_psp_projector_real(psp, i, l, r)

Evaluate the radial part of the `i`-th projector for angular momentum `l`
in real-space at the vector with modulus `r`.
"""
function eval_psp_projector_real end


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
function eval_psp_projector_fourier end

"""
    eval_psp_local_real(psp, r)

Evaluate the local part of the pseudopotential in real space.
"""
function eval_psp_local_real end

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
function eval_psp_local_fourier end

@doc raw"""
    eval_psp_energy_correction([T=Float64,] psp::NormConservingPsp)
    eval_psp_energy_correction([T=Float64,] element::Element)

Evaluate the energy correction to the Ewald electrostatic interaction energy per unit
of uniform negative charge. This is the correction required to account for the fact that
the Ewald expression assumes point-like nuclei and not nuclei of the shape induced by
the pseudopotential. The compensating background charge assumed for this expression is
scaled to ``1``. Therefore multiplying by the number of electrons and dividing by the unit
cell volume yields the energy correction per volume for the DFT simulation.

The energy correction is defined as the limit of the Fourier-transform of the local
potential as ``p \to 0``, using the same correction as in the Fourier-transform of the local
potential:
```math
\lim_{p \to 0} 4π N_{\rm elec} ∫_{ℝ_+} (V(r) - C(r)) \frac{\sin(p·r)}{p·r} r^2 dr + F[C(r)]
 = 4π N_{\rm elec} ∫_{ℝ_+} (V(r) + Z/r) r^2 dr.
```
where as discussed above the implementation is expected to return the result
for ``N_{\rm elec} = 1``.
"""
function eval_psp_energy_correction end

eval_psp_energy_correction(psp::NormConservingPsp) = eval_psp_energy_correction(Float64, psp)

#
# Function stubs for optional methods
#

"""
    eval_psp_valence_density_real(psp, r)

Evaluate the atomic valence charge density in real space.
"""
function eval_psp_valence_density_real end

@doc raw"""
    eval_psp_valence_density_fourier(psp, p)

Evaluate the atomic valence charge density in reciprocal space:
```math
\begin{aligned}
ρ_{\rm val}(p) &= ∫_{ℝ^3} ρ_{\rm val}(r) e^{-ip·r} dr \\
               &= 4π ∫_{ℝ_+} ρ_{\rm val}(r) \frac{\sin(p·r)}{ρ·r} r^2 dr.
\end{aligned}
```
"""
function eval_psp_valence_density_fourier end

"""
    eval_psp_core_density_real(psp, r)

Evaluate the atomic core charge density in real space.
"""
eval_psp_core_density_real(::NormConservingPsp, ::T) where {T <: Real} = zero(T)

@doc raw"""
    eval_psp_core_density_fourier(psp, p)

Evaluate the atomic core charge density in reciprocal space:
```math
\begin{aligned}
ρ_{\rm core}(p) &= ∫_{ℝ^3} ρ_{\rm core}(r) e^{-ip·r} dr \\
               &= 4π ∫_{ℝ_+} ρ_{\rm core}(r) \frac{\sin(p·r)}{ρ·r} r^2 dr.
\end{aligned}
```
"""
eval_psp_core_density_fourier(::NormConservingPsp, ::T) where {T <: Real} = zero(T)

eval_psp_core_kinetic_energy_density_real(::NormConservingPsp, ::T) where {T <: Real} = zero(T)
eval_psp_core_kinetic_energy_density_fourier(::NormConservingPsp, ::T) where {T <: Real} = zero(T)


"""
    eval_psp_pswfc_real(psp::PspUpf, i, l, r)

Evaluate the radial part of the `i`-th pseudo wave function of angular momentum `l`
at real space radius `r`.
"""
function eval_psp_pswfc_real end

"""
    eval_psp_pswfc_fourier(psp::PspUpf, i, l, p)

Evaluate the radial part of the `i`-th pseudo wave function of angular momentum `l`
at reciprocal space radius `z`.
"""
function eval_psp_pswfc_fourier end

#
# Other methods defined on a NormConservingPsp
#
#
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

count_n_pswfc_radial(psp::NormConservingPsp, l) = error("Pseudopotential $psp does not implement atomic wavefunctions.")

function count_n_pswfc_radial(psp::NormConservingPsp)
    sum(l -> count_n_pswfc_radial(psp, l), 0:psp.lmax; init=0)::Int 
end

count_n_pswfc(psp::NormConservingPsp, l) = count_n_pswfc_radial(psp, l) * (2l + 1)
function count_n_pswfc(psp::NormConservingPsp)
    sum(l -> count_n_pswfc(psp, l), 0:psp.lmax; init=0)::Int
end

function find_pswfc(psp::NormConservingPsp, label::String)
    for l = 0:psp.lmax, i = 1:count_n_pswfc_radial(psp, l)
        if pswfc_label(psp, i, l) == label
            return (; l, i)
        end
    end
    error("Could not find pseudo atomic orbital with label $label "
          * "in pseudopotential $(psp).")
end

#
# Provide dispatches for vector-valued inputs by calling an existing scalar version
# elementwise. Safe for GPUArray inputs. Performance critical methods should
# have their own GPU-optimized implementation instead of relying on this dispatch.
#
for fn in (:eval_psp_pswfc_fourier, :eval_psp_pswfc_real,
           :eval_psp_projector_real, :eval_psp_projector_fourier)
    @eval function $fn(psp::NormConservingPsp, i, l, ps::AbstractVector{<:Real})
        arch = architecture(ps)
        to_device(arch, map(p -> $fn(psp, i, l, p), to_cpu(ps)))
    end
end

for fn in (:eval_psp_local_real, :eval_psp_local_fourier,
           :eval_psp_valence_density_real, :eval_psp_valence_density_fourier,
           :eval_psp_core_density_real, :eval_psp_core_density_fourier,
           :eval_psp_core_kinetic_energy_density_real,
           :eval_psp_core_kinetic_energy_density_fourier)
    @eval function $fn(psp::NormConservingPsp, ps::AbstractVector{<:Real})
        arch = architecture(ps)
        to_device(arch, map(p -> $fn(psp, p), to_cpu(ps)))
    end
end
