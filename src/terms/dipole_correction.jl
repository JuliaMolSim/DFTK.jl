@doc raw"""
    DipoleCorrection(direction=3; discontinuity_position=0,
                     smoothening_width=1u"Å")

Dipole correction for asymmetric slabs as introduced by Bengtsson in
[Phys. Rev. B 59, 12301 (1999)](https://doi.org/10.1103/PhysRevB.59.12301).
The correction is applied along the selected lattice `direction`, which must be
orthogonal to the two other lattice vectors. `discontinuity_position` is the fractional
coordinate of the potential discontinuity along this direction and `smoothening_width`
is the total width of the cubic smoothing region. `smoothening_width` can be given in
atomic units or as a `Unitful` length. Add this term using `extra_terms`, for example
`model_DFT(...; extra_terms=[DipoleCorrection()])`.

The discontinuity and its smoothing region must lie in vacuum. The calculation must
be neutral and three-dimensional, and the smoothing region must be resolved by the
FFT grid. Let ``L`` be the length normal to the slab, ``A`` the lateral area and
``\Omega = A L`` the cell volume. With ``u(\mathbf r) = L\,S(\mathbf r)`` the signed
distance from the cell midpoint along `direction`, the dipole moment along `direction` is
```math
p = \int_\Omega \left(\rho(\mathbf r) - \sum_I Z_I \delta(\mathbf r-\mathbf R_I)\right)
            u(\mathbf r)\,dr
```
(with the sign following the positive electron density ``\rho``) and the correction reads
```math
E_\mathrm{dip} = \frac{2\pi}{\Omega}p^2, \qquad
V_\mathrm{dip} = \frac{4\pi}{\Omega}p\,u(\mathbf r).
```
If ``x`` measures the distance to the discontinuity in fractional coordinates,
then the sawtooth ``S`` is the linear function ``S(x) = x - 1/2`` (plus
additionally a smooth cubic to easy the discontinuity).

The implementation assumes that there are no symmetries in the direction along
which the sawtooth is applied. If this is not correct, manually disable
symmetry detection in DFTK (pass `symmetries=false` to the `Model`).
"""
struct DipoleCorrection
    direction::Int
    discontinuity_position::Float64
    smoothening_width::Float64
end
function DipoleCorrection(direction=3; discontinuity_position=0,
                          smoothening_width=austrip(1u"Å"))
    direction in 1:3 || throw(ArgumentError("direction must be 1, 2 or 3"))
    0 ≤ discontinuity_position ≤ 1 || throw(ArgumentError(
        "discontinuity_position must be a fractional coordinate in (0, 1)"))
    smoothening_width = austrip(smoothening_width)
    smoothening_width > 0 || throw(ArgumentError("smoothening_width must be positive"))
    DipoleCorrection(direction, discontinuity_position, smoothening_width)
end

struct TermDipoleCorrection{T, A} <: TermNonlinear
    sawtooth_scaled::A  # sawtooth S scaled by L at each spatial position
    ionic_dipole_moment::T
    ionic_dipole_derivative::Vector{Vec3{T}}
end

function TermDipoleCorrection(basis::PlaneWaveBasis{T},
                              correction::DipoleCorrection) where {T}
    model     = basis.model
    direction = correction.direction
    normal    = model.lattice[:, direction]
    position  = T(mod(correction.discontinuity_position, 1))

    model.n_dim == 3 ||
        throw(ArgumentError("DipoleCorrection requires a 3D model"))
    model.n_electrons == n_electrons_from_atoms(model.atoms) ||
        throw(ArgumentError("DipoleCorrection requires a neutral model"))
    correction.smoothening_width < norm(normal) || throw(ArgumentError(
        "The dipole-correction smoothing width must be smaller than the lattice period"))

    for lateral in (mod1(direction + 1, 3), mod1(direction + 2, 3))
        other = model.lattice[:, lateral]
        # A dimensionless tolerance allows small numerical non-orthogonality
        # in structures imported through AtomsBase.
        orthogonality = abs(dot(normal, other)) / (norm(normal) * norm(other))
        orthogonality ≤ sqrt(eps(value_type(T))) || throw(ArgumentError(
            "The dipole-correction direction must be orthogonal to the other lattice vectors"))
    end

    L = T(norm(normal))
    halfwidth = T(correction.smoothening_width) / (2L)

    sawtooth_scaled = map(to_cpu(r_vectors(basis))) do r
        L * _dipole_correction_sawtooth(r[direction] - position, halfwidth)
    end
    sawtooth_scaled = to_device(basis.architecture, sawtooth_scaled)

    ionic_dipole_moment = zero(T)
    ionic_dipole_derivative = Vec3{T}[]
    for (atom, r) in zip(model.atoms, model.positions)
        δr = r[direction] - position
        ionic_dipole_moment += charge_ionic(atom) * L * _dipole_correction_sawtooth(δr, halfwidth)

        derivative = zeros(T, 3)
        derivative[direction] = ForwardDiff.derivative(r[direction]) do x
            charge_ionic(atom) * L * _dipole_correction_sawtooth(x - position, halfwidth)
        end
        push!(ionic_dipole_derivative, derivative)
    end
    TermDipoleCorrection(sawtooth_scaled, ionic_dipole_moment,
                         ionic_dipole_derivative)
end
(t::DipoleCorrection)(basis) = TermDipoleCorrection(basis, t)

# Sawtooth function with slope 1 on the periodic domain (0, 1) with discontinuous
# jump at x=0 (i.e. x=1). A cubic is used to join the branches at x = halfwidth
# and x = (1 - halfwidth) with continuous value and derivaitve,
# i.e. the region of the cubic is x ∈ [0, -halfwidth] and x ∈ [1-halfwidth, 1].
function _dipole_correction_sawtooth(x, halfwidth)
    function cubic(x)
        u = x / halfwidth
        x + (u^3 - 3u) / 4
    end

    x = x - floor(Int, x)
    if x < halfwidth
        return cubic(x)
    elseif x ≥ 1 - halfwidth
        return -cubic(1 - x)
    else
        return x - one(x) / 2
    end
end

@timing "ene_ops: dipole correction" function ene_ops(term::TermDipoleCorrection,
                                                      basis::PlaneWaveBasis{T},
                                                      ψ, occupation;
                                                      ρ, kwargs...) where {T}
    Ω = basis.model.unit_cell_volume
    total_dipole_moment = (sum(total_density(ρ) .* term.sawtooth_scaled) * basis.dvol
                           - term.ionic_dipole_moment)
    E = 4T(π) / Ω * total_dipole_moment^2 / T(2)
    pot_real = 4T(π) / Ω * total_dipole_moment .* term.sawtooth_scaled
    ops = [RealSpaceMultiplication(basis, kpt, pot_real) for kpt in basis.kpoints]
    (; E, ops)
end

function compute_forces(term::TermDipoleCorrection, basis::PlaneWaveBasis{T},
                        ψ, occupation; ρ, kwargs...) where {T}
    Ω = basis.model.unit_cell_volume
    total_dipole_moment = (sum(total_density(ρ) .* term.sawtooth_scaled) * basis.dvol
                           - term.ionic_dipole_moment)
    map(basis.model.atoms, term.ionic_dipole_derivative) do atom, derivative
        4T(π) / Ω * total_dipole_moment * derivative
    end
end

function compute_kernel(term::TermDipoleCorrection, basis::PlaneWaveBasis{T};
                        kwargs...) where {T}
    Ω = basis.model.unit_cell_volume
    sawtooth_scaled = vec(term.sawtooth_scaled)
    K = 4T(π) / Ω * basis.dvol * (sawtooth_scaled * transpose(sawtooth_scaled))
    basis.model.n_spin_components == 1 ? K : [K K; K K]
end

function apply_kernel(term::TermDipoleCorrection, basis::PlaneWaveBasis{T},
                      δρ::AbstractArray; kwargs...) where {T}
    Ω = basis.model.unit_cell_volume
    δmoment = sum(total_density(δρ) .* term.sawtooth_scaled) * basis.dvol
    δV = zero(δρ)
    # All spin components see the same dipole potential.
    δV .= 4T(π) / Ω * δmoment .* term.sawtooth_scaled
    (; δVρ=δV)
end

function compute_dynmat(::TermDipoleCorrection, ::PlaneWaveBasis,
                        ψ, occupation; kwargs...)
    error("Phonons are not yet implemented for DipoleCorrection")
end
function compute_δHψ_αs(::TermDipoleCorrection, ::PlaneWaveBasis,
                         ψ, α, s, q; kwargs...)
    error("Phonons are not yet implemented for DipoleCorrection")
end
