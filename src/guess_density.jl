
"""
Generate a physically valid random density integrating to the given number of electrons.
"""
function random_density(basis::PlaneWaveBasis; n_electrons=basis.model.n_electrons)
    T = eltype(basis)
    ρtot  = rand(T, basis.fft_size)
    ρtot  = ρtot .* n_electrons ./ (sum(ρtot) * basis.dvol)  # Integration to n_electrons
    ρspin = nothing
    if basis.model.n_spin_components > 1
        ρspin = rand((-1, 1), basis.fft_size ) .* rand(T, basis.fft_size) .* ρtot
        @assert all(abs.(ρspin) .≤ ρtot)
    end
    ρ_from_total_and_spin(ρtot, ρspin)
end


@doc raw"""
    guess_density(basis, magnetic_moments)

Build a superposition of atomic densities (SAD) guess density.

We take for the guess density a gaussian centered around the atom, of
length specified by `atom_decay_length`, normalized to get the right number of electrons
```math
\hat{ρ}(G) = Z \exp\left(-(2π \text{length} |G|)^2\right)

When magnetic moments are provided, construct a symmetry-broken density guess.
The magnetic moments should be specified in units of ``μ_B``.
"""
function guess_density(basis::PlaneWaveBasis, magnetic_moments=[])
    guess_density(basis, basis.model.atoms, magnetic_moments)
end
@timing function guess_density(basis::PlaneWaveBasis{T}, atoms, magnetic_moments) where {T}
    ρtot = _guess_total_density(basis, atoms)
    if basis.model.n_spin_components == 1
        ρspin = nothing
    else
        ρspin = _guess_spin_density(basis, atoms, magnetic_moments)
    end

    ρ_from_total_and_spin(ρtot, ρspin)
end

function _guess_total_density(basis::PlaneWaveBasis{T}, atoms) where {T}
    # build ρtot
    gaussians_tot = [(T(n_elec_valence(spec)), T(atom_decay_length(spec)), pos)
                     for (spec, positions) in atoms for pos in positions]
    ρtot = gaussian_superposition(basis, gaussians_tot)
end

function _guess_spin_density(basis::PlaneWaveBasis{T}, atoms, magnetic_moments) where {T}
    model = basis.model
    if model.spin_polarization in (:none, :spinless)
        isempty(magnetic_moments) && return nothing
        error("Initial magnetic moments can only be used with collinear models.")
    end

    # If no magnetic moments start with a zero spin density
    all_magmoms = (normalize_magnetic_moment(magmom) for (_, magmoms) in magnetic_moments
                   for magmom in magmoms)
    if all(iszero, all_magmoms)
        @warn("Returning zero spin density guess, because no initial magnetization has " *
              "been specified in any of the given elements / atoms. Your SCF will likely " *
              "not converge to a spin-broken solution.")
        return zeros(T, basis.fft_size)
    end

    gaussians = Tuple{T, T, Vec3{T}}[]
    @assert length(magnetic_moments) == length(atoms)
    for (ispec, (spec, magmoms)) in enumerate(magnetic_moments)
        positions = atoms[ispec][2]
        @assert charge_nuclear(spec) == charge_nuclear(atoms[ispec][1])
        @assert length(magmoms) == length(positions)
        for (ipos, r) in enumerate(positions)
            magmom = Vec3{T}(normalize_magnetic_moment(magmoms[ipos]))
            iszero(magmom) && continue
            iszero(magmom[1:2]) || error("Non-collinear magnetization not yet implemented")

            magmom[3] ≤ n_elec_valence(spec) || error(
                "Magnetic moment $(magmom[3]) too large for element $(spec.symbol) with " *
                "only $(n_elec_valence(spec)) valence electrons."
            )
            push!(gaussians, (magmom[3], atom_decay_length(spec), r))
        end
    end
    gaussian_superposition(basis, gaussians)
end


@doc raw"""
Build a superposition of Gaussians as a guess for the density and magnetisation.
Expects a list of tuples `(coefficient, length, position)` for each of the Gaussian,
which follow the functional form
```math
\hat{ρ}(G) = \text{coefficient} \exp\left(-(2π \text{length} |G|)^2\right)
```
and are placed at `position` (in fractional coordinates).
"""
function gaussian_superposition(basis::PlaneWaveBasis{T}, gaussians) where {T}
    ρ = zeros(complex(T), basis.fft_size)
    isempty(gaussians) && return G_to_r(basis, ρ)

    # Fill ρ with the (unnormalized) Fourier transform, i.e. ∫ e^{-iGx} f(x) dx,
    # where f(x) is a weighted gaussian
    #
    # is formed from a superposition of atomic densities, each scaled by a prefactor
    for (iG, G) in enumerate(G_vectors(basis))
        Gsq = sum(abs2, basis.model.recip_lattice * G)
        for (coeff, decay_length, r) in gaussians
            form_factor::T = exp(-Gsq * T(decay_length)^2)
            ρ[iG] += T(coeff) * form_factor * cis(-2T(π) * dot(G, r))
        end
    end

    # projection in the normalized plane wave basis
    G_to_r(basis, ρ / sqrt(basis.model.unit_cell_volume))
end


@doc raw"""
Get the lengthscale of the valence density for an atom with `n_elec_core` core
and `n_elec_valence` valence electrons.
```
"""
function atom_decay_length(n_elec_core, n_elec_valence)
    # Adapted from ABINIT/src/32_util/m_atomdata.F90,
    # from which also the data has been taken.

    n_elec_valence = round(Int, n_elec_valence)
    if n_elec_valence == 0
        return 0.0
    end

    data = if n_elec_core < 0.5
        # Bare ions: Adjusted on 1H and 2He only
        [0.6, 0.4, 0.3, 0.25, 0.2]
    elseif n_elec_core < 2.5
        # 1s2 core: Adjusted on 3Li, 6C, 7N, and 8O
        [1.8, 1.4, 1.0, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3]
    elseif n_elec_core < 10.5
        # Ne core (1s2 2s2 2p6): Adjusted on 11na, 13al, 14si and 17cl
        [2.0, 1.6, 1.25, 1.1, 1.0, 0.9, 0.8, 0.7 , 0.7, 0.7, 0.6]
    elseif n_elec_core < 12.5
        # Mg core (1s2 2s2 2p6 3s2): Adjusted on 19k, and on n_elec_core==10
        [1.9, 1.5, 1.15, 1.0, 0.9, 0.8, 0.7, 0.6 , 0.6, 0.6, 0.5]
    elseif n_elec_core < 18.5
        # Ar core (Ne + 3s2 3p6): Adjusted on 20ca, 25mn and 30zn
        [2.0, 1.8, 1.5, 1.2, 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.65, 0.6]
    elseif n_elec_core < 28.5
        # Full 3rd shell core (Ar + 3d10): Adjusted on 31ga, 34se and 38sr
        [1.5, 1.25, 1.15, 1.05, 1.00, 0.95, 0.95, 0.9, 0.9, 0.85, 0.85, 0.80,
         0.8 , 0.75, 0.7]
    elseif n_elec_core < 36.5
        # Krypton core (Ar + 3d10 4s2 4p6): Adjusted on 39y, 42mo and 48cd
        [2.0, 2.00, 1.60, 1.40, 1.25, 1.10, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.7]
    else
        # For the remaining elements, consider a function of n_elec_valence only
        [2.0 , 2.00, 1.55, 1.25, 1.15, 1.10, 1.05, 1.0 , 0.95, 0.9, 0.85, 0.85, 0.8]
    end
    data[min(n_elec_valence, length(data))]
end
atom_decay_length(sp::Element) = atom_decay_length(n_elec_core(sp), n_elec_valence(sp))
