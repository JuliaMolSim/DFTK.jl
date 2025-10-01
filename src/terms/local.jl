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

    form_factors_cpu = zeros(T, length(norm_indices), length(basis.model.atom_groups))
    for (igroup, group) in enumerate(basis.model.atom_groups)
        element = basis.model.atoms[first(group)]
        atomic_local_inner_loop!(form_factors_cpu, norm_indices, igroup, element, basis.architecture)
    end

    form_factors = to_device(basis.architecture, form_factors_cpu)
    iG2ifnorm = to_device(basis.architecture, iG2ifnorm_cpu)
    (; form_factors, iG2ifnorm)
end
function atomic_local_inner_loop!(form_factors_cpu, norm_indices, igroup,
                        element::Element, arch::AbstractArchitecture)
    for (p, ifnorm) in norm_indices
        form_factors_cpu[ifnorm, igroup] = local_potential_fourier(element, p)
    end
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

    pot_fourier = reshape(pot, basis.fft_size)
    if iszero(q)
        enforce_real!(pot, basis)  # Symmetrize coeffs to have real iFFT
        return irfft(basis, pot_fourier)
    else
        return ifft(basis, pot_fourier)
    end
end
(::AtomicLocal)(basis::PlaneWaveBasis{T}) where {T} =
    TermAtomicLocal(compute_local_potential(basis))

function compute_forces(::TermAtomicLocal, basis::PlaneWaveBasis{T}, ψ, occupation;
                        ρ, kwargs...) where {T}
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
