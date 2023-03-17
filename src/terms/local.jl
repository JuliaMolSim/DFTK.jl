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
function (::AtomicLocal)(basis::PlaneWaveBasis{T}) where {T}
    # pot_fourier is <e_G|V|e_G'> expanded in a basis of e_{G-G'}
    # Since V is a sum of radial functions located at atomic
    # positions, this involves a form factor (`local_potential_fourier`)
    # and a structure factor e^{-i G·r}
    model = basis.model
    G_cart = to_cpu(G_vectors_cart(basis))
    # TODO Bring G_cart on the CPU for compatibility with the pseudopotentials which
    #      are not isbits ... might be able to solve this by restructuring the loop


    # Pre-compute the form factors at unique values of |G| to speed up
    # the potential Fourier transform (by a lot). Using a hash map gives O(1)
    # lookup.
    form_factors = IdDict{Tuple{Int,T},T}()  # IdDict for Dual compatability
    for G in G_cart
        q = norm(G)
        for (igroup, group) in enumerate(model.atom_groups)
            if !haskey(form_factors, (igroup, q))
                element = model.atoms[first(group)]
                form_factors[(igroup, q)] = local_potential_fourier(element, q)
            end
        end
    end

    Gs = to_cpu(G_vectors(basis))  # TODO Again for GPU compatibility
    pot_fourier = map(enumerate(Gs)) do (iG, G)
        q = norm(G_cart[iG])
        pot = sum(enumerate(model.atom_groups)) do (igroup, group)
            structure_factor = sum(r -> cis2pi(-dot(G, r)), @view model.positions[group])
            form_factors[(igroup, q)] * structure_factor
        end
        pot / sqrt(model.unit_cell_volume)
    end

    enforce_real!(basis, pot_fourier)  # Symmetrize Fourier coeffs to have real iFFT
    pot_real = irfft(basis, to_device(basis.architecture, pot_fourier))

    TermAtomicLocal(pot_real)
end

@timing "forces: local" function compute_forces(::TermAtomicLocal, basis::PlaneWaveBasis{T},
                                                ψ, occupation; ρ, q=zeros(3),
                                                kwargs...) where {T}

    cast = iszero(q) ? real : complex
    S = promote_type(T, cast(eltype(ψ[1])))
    model = basis.model
    recip_lattice = model.recip_lattice
    ρ_fourier = fft(basis, total_density(ρ))

    # energy = sum of form_factor(G) * struct_factor(G) * rho(G)
    # where struct_factor(G) = e^{-i G·r}
    forces = [zero(Vec3{S}) for _ in 1:length(model.positions)]
    for group in model.atom_groups
        element = model.atoms[first(group)]
        form_factors = [S(local_potential_fourier(element, norm(recip_lattice * (G + q))))
                        for G in G_vectors(basis)]
        for idx in group
            r = model.positions[idx]
            forces[idx] = _force_local_internal(basis, ρ_fourier, form_factors, r; q)
        end
    end
    forces
end

# function barrier to work around various type instabilities
function _force_local_internal(basis::PlaneWaveBasis{T}, ρ_fourier, form_factors, r;
                               q=zero(Vec3{T})) where {T}
    cast = iszero(q) ? real : complex
    S = cast(eltype(ρ_fourier))
    f = zero(Vec3{S})
    for (iG, G) in enumerate(G_vectors(basis))
        f -= cast(conj(ρ_fourier[iG])
                      .* form_factors[iG]
                      .* cis2pi(-dot(G + q, r))
                      .* (-2T(π) .* (G + q) .* im)
                      ./ sqrt(basis.model.unit_cell_volume))
    end
    f
end

function compute_δV(basis::PlaneWaveBasis{T}, pσ, γ, pg; q=zero(Vec3{T})) where {T}
    model = basis.model
    # pot_fourier is <e_G|V|e_G'> expanded in a basis of e_{G-G'}
    # Since V is a sum of radial functions located at atomic
    # positions, this involves a form factor (`local_potential_fourier`)
    # and a structure factor e^{-i Gr}

    pot_fourier = map(G_vectors(basis)) do G
        pot = let
            element = model.atoms[pg]
            form_factor::T = local_potential_fourier(element, norm(model.recip_lattice * (G + q)))
            form_factor * (-2T(π) * im * (G[γ] + q[γ])) * cis2pi(-dot(G + q, pσ))
        end
        pot / sqrt(model.unit_cell_volume)
    end

    ifft(basis, pot_fourier)
end

function compute_δ²V(basis::PlaneWaveBasis{T}, σ, τ, γ, η) where {T}
    model = basis.model
    pσ = model.positions[σ]
    # pot_fourier is <e_G|V|e_G'> expanded in a basis of e_{G-G'}
    # Since V is a sum of radial functions located at atomic
    # positions, this involves a form factor (`local_potential_fourier`)
    # and a structure factor e^{-i Gr}

    σ ≢ τ && return zeros(complex(T), basis.fft_size...)
    pot_fourier = map(G_vectors(basis)) do G
        pot = let
            element = model.atoms[σ]
            local_potential_fourier(element, norm(model.recip_lattice * G))
        end
        pot *= (-2T(π) * im * G[γ]) .* (-2T(π) * im * G[η]) * cis2pi(-dot(G, pσ))
        pot / sqrt(model.unit_cell_volume)
    end

    ifft(basis, pot_fourier)
end

function compute_δ²V_fourier(basis::PlaneWaveBasis{T}, σ, τ, γ, η) where {T}
    model = basis.model
    pσ = model.positions[σ]
    # pot_fourier is <e_G|V|e_G'> expanded in a basis of e_{G-G'}
    # Since V is a sum of radial functions located at atomic
    # positions, this involves a form factor (`local_potential_fourier`)
    # and a structure factor e^{-i Gr}

    σ ≢ τ && return zeros(complex(T), basis.fft_size...)
    pot_fourier = map(G_vectors(basis)) do G
        pot = let
            element = model.atoms[σ]
            local_potential_fourier(element, norm(model.recip_lattice * G))
        end
        pot *= (-2T(π) * im * G[γ]) .* (-2T(π) * im * G[η]) * cis2pi(-dot(G, pσ))
        pot / sqrt(model.unit_cell_volume)
    end

    pot_fourier
end

function compute_δV(::TermAtomicLocal, basis::PlaneWaveBasis{T}; q=zero(Vec3{T})) where {T}
    S = complex(T)
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim
    spins = model.n_spin_components

    δV = Array{Array{S, 4}, 2}(undef, n_dim, n_atoms)
    for (τ, position) in enumerate(positions)
        for γ in 1:n_dim
            δV_τγ = compute_δV(basis, position, γ, τ; q)
            temp = Array{S, 4}(undef, (basis.fft_size..., spins))
            for spin in 1:spins
                temp[:, :, :, spin] = δV_τγ
            end
            δV[γ, τ] = temp
        end
    end
    δV
end

function compute_∫δρδV(term::TermAtomicLocal, scfres::NamedTuple, δρs, δψs, δoccupations;
                       q=zero(Vec3{eltype(scfres.basis)}))
    basis = scfres.basis
    S = complex(eltype(basis))
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim

    ∫δρδV_term = zeros(S, (n_dim, n_atoms, n_dim, n_atoms))
    for τ in 1:n_atoms
        for γ in 1:n_dim
            ∫δρδV_τγ = -compute_forces(term, basis, scfres.ψ, scfres.occupation;
                                    ρ=δρs[γ, τ], δψ=δψs[γ, τ], δoccupation=δoccupations[γ, τ],
                                    q, qpt=q)
            ∫δρδV_term[:, :, γ, τ] .+= hcat(∫δρδV_τγ...)[1:n_dim, :]
        end
    end

    reshape(∫δρδV_term, n_dim*n_atoms, n_dim*n_atoms)
end

function compute_δ²V(::TermAtomicLocal, basis::PlaneWaveBasis{T}) where {T}
    S = complex(T)
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim

    δ²V = zeros(S, (basis.fft_size..., n_dim, n_atoms, n_dim, n_atoms))
    for τ in 1:n_atoms
        for γ in 1:n_dim
            for σ in 1:n_atoms
                for η in 1:n_dim
                    δ²V_στγη = compute_δ²V(basis, σ, τ, γ, η)
                    δ²V[:, :, :, η, σ, γ, τ] = δ²V_στγη
                end
            end
        end
    end
    δ²V
end

function compute_∫ρδ²V(term::TermAtomicLocal, scfres::NamedTuple)
    basis = scfres.basis
    S = complex(eltype(basis))
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim
    ρ_fourier = fft(basis, total_density(scfres.ρ))


    ∫ρδ²V_term = zeros(S, (n_dim, n_atoms, n_dim, n_atoms))
    δ²V = compute_δ²V(term, basis)
    for τ in 1:n_atoms
        for σ in 1:n_atoms
            for γ in 1:n_dim
                for η in 1:n_dim
                    δ²V_fourier = fft(basis, δ²V[:, :, :, η, σ, γ, τ])
                    ∫ρδ²V_τσγ = sum(conj(ρ_fourier) .* δ²V_fourier)
                    ∫ρδ²V_term[η, σ, γ, τ] = ∫ρδ²V_τσγ
                end
            end
        end
    end
    reshape(∫ρδ²V_term, n_dim*n_atoms, n_dim*n_atoms)
end

function compute_dynmat(term::TermAtomicLocal, scfres::NamedTuple; δρs, δψs, δoccupations,
                        q=zero(Vec3{eltype(scfres.basis)}))
    ∫δρδV = compute_∫δρδV(term, scfres, δρs, δψs, δoccupations; q)
    ∫ρδ²V = compute_∫ρδ²V(term, scfres)
    return ∫δρδV + ∫ρδ²V
end

function compute_δHψ(term::TermAtomicLocal, scfres::NamedTuple;
                     q=zero(Vec3{eltype(scfres.basis)}))
    basis = scfres.basis
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim

    δV = compute_δV(term, basis; q)
    δHψ = [zero.(scfres.ψ) for _ in 1:n_dim, _ in 1:n_atoms]
    for τ in 1:n_atoms
        for γ in 1:n_dim
            δHψ_τγ = multiply_by_δV_expiqr_fourier(basis, -q, scfres.ψ, δV[γ, τ])
            δHψ[γ, τ] .+= δHψ_τγ
       end
   end
   δHψ
end
