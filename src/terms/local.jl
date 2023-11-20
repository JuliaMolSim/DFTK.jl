## Local potentials. Can be provided from external potentials, or from `model.atoms`.

# A local potential term. Must have the field `potential_values`, storing the
# potential in real space on the grid. If the potential is different in the α and β
# components then it should be a 4d-array with the last axis running over the
# two spin components.
abstract type TermLocalPotential <: Term end

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
External potential from an analytic function `V` (in cartesian coordinates).
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
G is passed in cartesian coordinates
"""
struct ExternalFromFourier
    potential::Function
end
function (external::ExternalFromFourier)(basis::PlaneWaveBasis{T}) where {T}
    unit_cell_volume = basis.model.unit_cell_volume
    pot_fourier = map(G_vectors_cart(basis)) do G
        convert_dual(complex(T), external.potential(G) / sqrt(unit_cell_volume))
    end
    enforce_real!(basis, pot_fourier)  # Symmetrize Fourier coeffs to have real iFFT
    TermExternal(irfft(basis, pot_fourier))
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
    model = basis.model
    Gs_cart = to_cpu([model.recip_lattice * (G + q) for G in G_vectors(basis)])
    # TODO Bring Gs_cart on the CPU for compatibility with the pseudopotentials which
    #      are not isbits ... might be able to solve this by restructuring the loop

    # Pre-compute the form factors at unique values of |G| to speed up
    # the potential Fourier transform (by a lot). Using a hash map gives O(1)
    # lookup.
    form_factors = IdDict{Tuple{Int,T},T}()  # IdDict for Dual compatability
    for G in Gs_cart
        p = norm(G)
        for (igroup, group) in enumerate(model.atom_groups)
            if !haskey(form_factors, (igroup, p))
                element = model.atoms[first(group)]
                form_factors[(igroup, p)] = local_potential_fourier(element, p)
            end
        end
    end

    Gs = to_cpu([G + q for G in G_vectors(basis)])  # TODO Again for GPU compatibility
    pot_fourier = map(enumerate(Gs)) do (iG, G)
        p = norm(Gs_cart[iG])
        pot = sum(enumerate(model.atom_groups)) do (igroup, group)
            structure_factor = sum(r -> cis2pi(-dot(G, r)), @view positions[group])
            form_factors[(igroup, p)] * structure_factor
        end
        pot / sqrt(model.unit_cell_volume)
    end

    iszero(q) && enforce_real!(basis, pot_fourier)  # Symmetrize coeffs to have real iFFT
    ifft(basis, to_device(basis.architecture, pot_fourier); real=iszero(q))
end
(::AtomicLocal)(basis::PlaneWaveBasis) = TermAtomicLocal(compute_local_potential(basis))

@timing "forces: local" function compute_forces(::TermAtomicLocal, basis::PlaneWaveBasis{T},
                                                ψ, occupation; ρ, q=zeros(3),
                                                kwargs...) where {T}

    S = promote_type(T, eltype(ψ[1]))
    model = basis.model
    recip_lattice = model.recip_lattice
    ρ_fourier = fft(basis, total_density(ρ))

    # energy = sum of form_factor(G) * struct_factor(G) * rho(G)
    # where struct_factor(G) = e^{-i G·r}
    forces = [zero(Vec3{S}) for _ = 1:length(model.positions)]
    for group in model.atom_groups
        element = model.atoms[first(group)]
        form_factors = [S(local_potential_fourier(element, norm(recip_lattice * (G + q))))
                        for G in G_vectors(basis)]
        for idx in group
            r = model.positions[idx]
            forces[idx] = _force_local_internal(basis, ρ_fourier, form_factors, r; q)
        end
    end
    iszero(q) ? real(forces) : forces
end

# function barrier to work around various type instabilities
function _force_local_internal(basis::PlaneWaveBasis{T}, ρ_fourier, form_factors, r;
                               q=zero(Vec3{T})) where {T}
    S = eltype(ρ_fourier)
    f = zero(Vec3{S})
    for (iG, G) in enumerate(G_vectors(basis))
        f -= (conj(ρ_fourier[iG])
                  .* form_factors[iG]
                  .* cis2pi(-dot(G + q, r))
                  .* (-2T(π)) .* (G + q) .* im
                  ./ sqrt(basis.model.unit_cell_volume))
    end
    f
end

# Phonon: Perturbation of the local potential using AD with respect to a displacement on the
# direction γ of the atom τ.
function compute_δV(::TermAtomicLocal, basis::PlaneWaveBasis{T}, γ, τ;
                    q=zero(Vec3{T}), positions=basis.model.positions) where {T}
    displacement = zero.(positions)
    displacement[τ] = setindex(displacement[τ], one(T), γ)
    ForwardDiff.derivative(zero(T)) do ε
        positions = ε*displacement .+ positions
        compute_local_potential(basis; q, positions)
    end
end

# Phonon: Second-order perturbation of the local potential using AD with respect to
# a displacement on the directions η and γ of the atoms σ et τ.
function compute_δ²V(term::TermAtomicLocal, basis::PlaneWaveBasis{T}, η, σ, γ, τ) where {T}
    model = basis.model

    displacement = zero.(model.positions)
    displacement[σ] = setindex(displacement[σ], one(T), η)
    ForwardDiff.derivative(zero(T)) do ε
        positions = ε*displacement .+ model.positions
        compute_δV(term, basis, γ, τ; positions)
    end
end

function compute_dynmat(term::TermAtomicLocal, basis::PlaneWaveBasis{T}, ψ, occupation; ρ,
                        δρs, δψs, δoccupations, q=zero(Vec3{T})) where {T}
    S = complex(eltype(basis))
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim

    ∫δρδV = zeros(S, 3, n_atoms, 3, n_atoms)
    for τ in 1:n_atoms
        for γ in 1:n_dim
            ∫δρδV_τγ = -compute_forces(term, basis, ψ, occupation; ρ=δρs[γ, τ],
                                       δψ=δψs[γ, τ], δoccupation=δoccupations[γ, τ], q)
            ∫δρδV[:, :, γ, τ] .+= hcat(∫δρδV_τγ...)
        end
    end

    ∫ρδ²V = zeros(S, 3, n_atoms, 3, n_atoms)
    ρ_fourier = fft(basis, total_density(ρ))
    for τ in 1:n_atoms
        for γ in 1:n_dim
            for σ in 1:n_atoms
                for η in 1:n_dim
                    δ²V_fourier = fft(basis, compute_δ²V(term, basis, η, σ, γ, τ))
                    ∫ρδ²V[η, σ, γ, τ] = sum(conj(ρ_fourier) .* δ²V_fourier)
                end
            end
        end
    end


    ∫δρδV + ∫ρδ²V
end

function compute_δHψ(term::TermAtomicLocal, basis::PlaneWaveBasis{T}, ψ, occupation;
                     q=zero(Vec3{T})) where {T}
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim

    δHψ = [zero.(ψ) for _ in 1:3, _ in 1:n_atoms]
    for τ in 1:n_atoms
        for γ in 1:n_dim
            δV_τγ = compute_δV(term, basis, γ, τ; q)
            δHψ_τγ = compute_δVψk(basis, q, ψ, cat(δV_τγ, δV_τγ; dims=4))
            δHψ[γ, τ] .+= δHψ_τγ
       end
   end
   δHψ
end
