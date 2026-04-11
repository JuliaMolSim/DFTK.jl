## Local potentials. Can be provided from external potentials, or from `model.atoms`.

# A local potential term. Must have the field `potential_values`, storing the
# potential in real space on the grid. If the potential is different in the α and β
# components then it should be a 4d-array with the last axis running over the
# two spin components.
abstract type TermLocalPotential <: TermLinear end

@timing "ene_ops: local" function ene_ops(term::TermLocalPotential,
                                          basis::PlaneWaveBasis{T}, ψ, occupation;
                                          kwargs...) where {T}
    potview(data, spin) = ndims(data) == 4 ? (@view data[:, :, :, spin]) : data
    ops = [RealSpaceMultiplication(basis, kpt, potview(term.potential_values, kpt.spin))
           for kpt in basis.kpoints]
    if :ρ in keys(kwargs)
        E = sum(total_density(kwargs[:ρ]) .* term.potential_values) * basis.dvol
    else
        E = T(Inf)
    end

    (; E, ops)
end

## External potentials

struct TermExternal <: TermLocalPotential
    potential_values::AbstractArray
end

"""
External potential given as values.
"""
struct ExternalFromValues
    potential_values::AbstractArray
end
function (external::ExternalFromValues)(basis::PlaneWaveBasis{T}) where {T}
    # TODO Could do interpolation here
    @assert size(external.potential_values) == basis.fft_size
    TermExternal(convert_dual.(T, external.potential_values))
end

"""
External potential from an analytic function `V` (in Cartesian coordinates).
No low-pass filtering is performed.
"""
struct ExternalFromReal
    potential::Function
end

function (external::ExternalFromReal)(basis::PlaneWaveBasis{T}) where {T}
    pot_real = external.potential.(r_vectors_cart(basis))
    TermExternal(convert_dual.(T, pot_real))
end

"""
External potential from the (unnormalized) Fourier coefficients `V(G)`
G is passed in Cartesian coordinates
"""
struct ExternalFromFourier
    potential::Function
end
function (external::ExternalFromFourier)(basis::PlaneWaveBasis{T}) where {T}
    unit_cell_volume = basis.model.unit_cell_volume
    pot_fourier = map(G_vectors_cart(basis)) do G
        convert_dual(complex(T), external.potential(G) / sqrt(unit_cell_volume))
    end
    enforce_real!(pot_fourier, basis)  # Symmetrize Fourier coeffs to have real iFFT
    TermExternal(irfft(basis, pot_fourier))
end

"""
Returns the form factors at unique values of |G + q| (in Cartesian coordinates).
Additionally, returns a mapping from any G index to the corresponding entry in the form_factors array.
"""
function atomic_local_form_factors(basis::PlaneWaveBasis{T}; q=zero(Vec3{T})) where{T}
    Gqs_cart = [basis.model.recip_lattice * (G + q) for G in to_cpu(G_vectors(basis))]

    iG2ifnorm_cpu = zeros(Int, length(Gqs_cart))
    norm_indices = IdDict{T, Int}()
    for (iG, G) in enumerate(Gqs_cart)
        p = norm(G)
        iG2ifnorm_cpu[iG] = get!(norm_indices, p, length(norm_indices) + 1)
    end
    iG2ifnorm = to_device(basis.architecture, iG2ifnorm_cpu)

    ni_pairs = collect(pairs(norm_indices))
    ps = to_device(basis.architecture, first.(ni_pairs))
    indices = to_device(basis.architecture, last.(ni_pairs))

    form_factors = similar(ps, length(norm_indices), length(basis.model.atom_groups))
    for (igroup, group) in enumerate(basis.model.atom_groups)
        element = basis.model.atoms[first(group)]
        @inbounds form_factors[indices, igroup] .= local_potential_fourier(element, ps)
    end

    (; form_factors, iG2ifnorm)
end

## Atomic local potential
struct TermAtomicLocal{AT} <: TermLocalPotential
    potential_values::AT
end

"""
Atomic local potential defined by `model.atoms`.
"""
struct AtomicLocal end
function compute_local_potential(basis::PlaneWaveBasis{T}; positions=basis.model.positions,
                                 q=zero(Vec3{T})) where {T}
    # pot_fourier is <e_G|V|e_G'> expanded in a basis of e_{G-G'}
    # Since V is a sum of radial functions located at atomic
    # positions, this involves a form factor (`local_potential_fourier`)
    # and a structure factor e^{-i G·r}
    form_factors, iG2ifnorm = atomic_local_form_factors(basis; q)
    Gqs = map(G -> G+q, G_vectors(basis))

    # Pre-allocation of large arrays for GPU efficiency
    Tpot = promote_type(eltype(form_factors), eltype(eltype(positions)))
    pot = to_device(basis.architecture, zeros(Complex{Tpot}, length(Gqs)))
    pot_tmp = similar(pot)
    indices = to_device(basis.architecture, collect(1:length(Gqs)))

    for (igroup, group) in enumerate(basis.model.atom_groups)
        for r in positions[group]
            ff_group = @view form_factors[:, igroup]
            map!(iG -> cis2pi(-dot(Gqs[iG], r)) * ff_group[iG2ifnorm[iG]], pot_tmp, indices)
            pot .+= pot_tmp ./ sqrt(basis.model.unit_cell_volume)
        end
    end

    # Apply the Rozzi truncated Coulomb correction to the long-range tail of
    # each atom's local potential whenever the electrostatics is not fully
    # periodic. This replaces -Z_a/r by -Z_a · v_c(r) at long range.
    if !is_fully_periodic_electrostatics(basis.model)
        add_truncated_coulomb_ionic_correction!(pot, basis, positions, q)
    end

    pot_fourier = reshape(pot, basis.fft_size)
    if iszero(q)
        enforce_real!(pot, basis)  # Symmetrize coeffs to have real iFFT
        return irfft(basis, pot_fourier)
    else
        return ifft(basis, pot_fourier)
    end
end

"""
Add the Rozzi truncated Coulomb correction to a fully-assembled periodic
atomic local potential `pot` (flattened Fourier coefficients in `sqrt(Ω)`
normalisation). This in-place modifies `pot` so that each atom's long-range
``-Z_a/r`` tail is replaced by the truncated Coulomb ``-Z_a · v_c(r)`` tail.

Only `q = 0` is currently supported.
"""
function add_truncated_coulomb_ionic_correction!(pot, basis::PlaneWaveBasis{T},
                                                 positions, q) where {T}
    model = basis.model
    iszero(q) || error("Truncated Coulomb corrections with q ≠ 0 are not yet " *
                       "supported (no phonon support yet).")

    # G_vectors returns a 3D array; flatten to 1D so we can index linearly.
    Gs_cpu = vec(to_cpu(G_vectors(basis)))  # Vector{Vec3{Int}}, length = prod(fft_size)
    recip_lattice = model.recip_lattice

    # Per-G correction factor Z_a * (4π/|G|² - v_c(G)), evaluated for each
    # atomic group (all atoms in a group share the same ionic charge). At G=0
    # we substitute the correct finite limit: V_short_a(0) - Z_a · v_c(0),
    # with V_short_a(0) = eval_psp_energy_correction(T, el_a).
    vc0 = truncated_coulomb_fourier(zero(Vec3{T}), model)
    Ω = model.unit_cell_volume

    # Precompute per-G correction factor (same for all atoms of the same group)
    # to avoid redundant kernel evaluations. correction_cpu is 1D, same length as Gs_cpu.
    n_G = length(Gs_cpu)
    correction_cpu = zeros(Complex{T}, n_G)

    for (igroup, group) in enumerate(model.atom_groups)
        element = model.atoms[first(group)]
        Za = T(charge_ionic(element))
        Vshort0 = T(eval_psp_energy_correction(T, element))

        # Per-G correction form factor for this element group
        fG_group = Vector{T}(undef, n_G)
        for iG = 1:n_G
            Gcart = recip_lattice * Gs_cpu[iG]
            Gsq = sum(abs2, Gcart)
            if iszero(Gsq)
                fG_group[iG] = Vshort0 - Za * vc0
            else
                vc = truncated_coulomb_fourier(Gcart, model)
                fG_group[iG] = Za * (4T(π) / Gsq - vc)
            end
        end

        # Accumulate structure-factor-weighted correction for each atom
        for idx in group
            r = positions[idx]
            for iG = 1:n_G
                G = Gs_cpu[iG]
                correction_cpu[iG] += cis2pi(-dot(G, r)) * fG_group[iG] / sqrt(Ω)
            end
        end
    end

    pot .+= to_device(basis.architecture, correction_cpu)
    pot
end
(::AtomicLocal)(basis::PlaneWaveBasis{T}) where {T} =
    TermAtomicLocal(compute_local_potential(basis))

function compute_forces(::TermAtomicLocal, basis::PlaneWaveBasis{T}, ψ, occupation;
                        ρ, kwargs...) where {T}
    # TODO: for non-periodic electrostatics the correction term added in
    # compute_local_potential (add_truncated_coulomb_ionic_correction!) also
    # contributes to the forces. This derivative is not yet implemented and
    # forces will be approximate in the truncated-Coulomb case.
    S = promote_type(T, real(eltype(ψ[1])))
    forces_local(S, basis, ρ, zero(Vec3{T}))
end
@timing "forces: local" function forces_local(S, basis::PlaneWaveBasis{T}, ρ, q) where {T}
    model = basis.model
    real_ifSreal = S <: Real ? real : identity

    form_factors, iG2ifnorm = atomic_local_form_factors(basis; q)

    Gqs = map(G -> G+q, G_vectors(basis))
    ρ_fourier = reshape(fft(basis, total_density(ρ)), length(Gqs))

    # Pre-allocation of large arrays for GPU efficiency
    indices = to_device(basis.architecture, collect(1:length(Gqs)))
    ρ_pot = similar(ρ_fourier)

    # energy = sum of form_factor(G) * struct_factor(G) * rho(G)
    # where struct_factor(G) = e^{-i G·r}
    forces = Vec3{S}[zero(Vec3{S}) for _ = 1:length(model.positions)]
    for (igroup, group) in enumerate(model.atom_groups)
        for idx in group
            r = model.positions[idx]

            ff_group = @view form_factors[:, igroup]
            map!(ρ_pot, indices) do iG
                cis2pi(-dot(Gqs[iG], r)) * conj(ρ_fourier[iG]) * ff_group[iG2ifnorm[iG]]
            end

            forces[idx] += map(1:3) do α
                tmp = sum(indices) do iG
                    -2π*im*Gqs[iG][α] * ρ_pot[iG]
                end
                -real_ifSreal(tmp / sqrt(model.unit_cell_volume))
            end
        end
    end
    forces
end

@views function compute_dynmat(::TermAtomicLocal, basis::PlaneWaveBasis{T}, ψ, occupation;
                               ρ, δρs, q=zero(Vec3{T}), kwargs...) where {T}
    !is_fully_periodic_electrostatics(basis.model) && error(
        "Phonon dynamical matrices with truncated Coulomb electrostatics " *
        "are not yet implemented.")
    S = complex(T)
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim

    # Two contributions: dynmat_δH and dynmat_δ²H

    # dynmat_δH, which is ∫δρδV.
    dynmat_δH = zeros(S, 3, n_atoms, 3, n_atoms)
    for s = 1:n_atoms, α = 1:n_dim
        dynmat_δH[:, :, α, s] .-= stack(forces_local(S, basis, δρs[α, s], q))
    end

    # dynmat_δ²H, which is ∫ρδ²V.
    dynmat_δ²H = zeros(S, 3, n_atoms, 3, n_atoms)
    ρ_fourier = fft(basis, total_density(ρ))
    δ²V_fourier = similar(ρ_fourier)
    for s = 1:n_atoms, α = 1:n_dim, β = 1:n_dim  # zero if s ≠ t
        δ²V = derivative_wrt_αs(basis.model.positions, β, s) do positions_βs
            derivative_wrt_αs(positions_βs, α, s) do positions_βsαs
                compute_local_potential(basis; positions=positions_βsαs)
            end
        end
        dynmat_δ²H[β, s, α, s] += sum(conj(ρ_fourier) .* fft!(δ²V_fourier, basis, δ²V))
    end

    dynmat_δH + dynmat_δ²H
end

# δH is the perturbation of the local potential due to a position displacement e^{iq·r} of
# the α coordinate of atom s.
function compute_δHψ_αs(::TermAtomicLocal, basis::PlaneWaveBasis, ψ, α, s, q)
    δV_αs = similar(ψ[1], basis.fft_size..., basis.model.n_spin_components)
    # Perturbation of the local potential with respect to a displacement on the direction α
    # of the atom s. All spin components get the same.
    δV_αs .= derivative_wrt_αs(basis.model.positions, α, s) do positions_αs
        compute_local_potential(basis; q, positions=positions_αs)
    end
    multiply_ψ_by_blochwave(basis, ψ, δV_αs, q)
end
