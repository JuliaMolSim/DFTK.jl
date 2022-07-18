
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
    guess_density(basis, magnetic_moments=[])
    guess_density(basis, system)

Build a superposition of atomic densities (SAD) guess density.

We take for the guess density a Gaussian centered around the atom, of
length specified by `atom_decay_length`, normalized to get the right number of electrons
```math
\hat{ρ}(G) = Z \exp\left(-(2π \text{length} |G|)^2\right)
```
When magnetic moments are provided, construct a symmetry-broken density guess.
The magnetic moments should be specified in units of ``μ_B``.
"""
function guess_density(basis::PlaneWaveBasis, magnetic_moments=[])
    guess_density(basis, basis.model.atoms, basis.model.positions, magnetic_moments)
end
function guess_density(basis::PlaneWaveBasis, system::AbstractSystem)
    parsed = parse_system(system)
    guess_density(basis, parsed.atoms, parsed.positions, parsed.magnetic_moments)
end
@timing function guess_density(basis::PlaneWaveBasis, atoms, positions, magnetic_moments)
    ρtot = _guess_total_density(basis, atoms, positions)
    if basis.model.n_spin_components == 1
        ρspin = nothing
    else
        ρspin = _guess_spin_density(basis, atoms, positions, magnetic_moments)
    end
    ρ_from_total_and_spin(ρtot, ρspin)
end

function _guess_total_density(basis::PlaneWaveBasis{T}, atoms, positions) where {T}
    @assert length(atoms) == length(positions)
    gaussians_tot = [(T(n_elec_valence(atom))::T, T(atom_decay_length(atom))::T, position)
                     for (atom, position) in zip(atoms, positions)]
    ρtot = gaussian_superposition(basis, gaussians_tot)
end

function _guess_spin_density(basis::PlaneWaveBasis{T}, atoms, positions, magnetic_moments) where {T}
    model = basis.model
    if model.spin_polarization in (:none, :spinless)
        isempty(magnetic_moments) && return nothing
        error("Initial magnetic moments can only be used with collinear models.")
    end

    # If no magnetic moments start with a zero spin density
    magmoms = Vec3{T}[normalize_magnetic_moment(magmom) for magmom in magnetic_moments]
    if all(iszero, magmoms)
        @warn("Returning zero spin density guess, because no initial magnetization has " *
              "been specified in any of the given elements / atoms. Your SCF will likely " *
              "not converge to a spin-broken solution.")
        array_type = typeof(similar(basis.G_vectors, T, basis.fft_size))
        return convert(array_type,zeros(T, basis.fft_size))
    end

    @assert length(magmoms) == length(atoms) == length(positions)
    gaussians = map(zip(atoms, positions, magmoms)) do (atom, position, magmom)
        iszero(magmom[1:2]) || error("Non-collinear magnetization not yet implemented")
        magmom[3] ≤ n_elec_valence(atom) || error(
            "Magnetic moment $(magmom[3]) too large for element $(atomic_symbol(atom)) " *
            "with only $(n_elec_valence(atom)) valence electrons."
        )
        magmom[3], T(atom_decay_length(atom))::T, position
    end
    gaussians = filter(g -> !iszero(g[1]), gaussians)
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
    ρ = deepcopy(basis.G_vectors)
    #These copies are required so that recip_lattice and gaussians are isbits (GPU compatibility)
    recip_lattice = basis.model.recip_lattice
    gaussians = SVector{size(gaussians)[1]}(gaussians)
    function build_ρ(G)
        Gsq = sum(abs2, recip_lattice * G)
        res = zero(complex(T))
        for (coeff, decay_length, r) in gaussians
            form_factor::T = exp(-Gsq * T(decay_length)^2)
            res += T(coeff) * form_factor* cis2pi(-dot(G, r))
        end
        res
    end
    ρ = map(build_ρ, ρ)/ sqrt(basis.model.unit_cell_volume) #Can't use map! as we are converting an array of Vec3 to an array of complex

    # projection in the normalized plane wave basis
    G_to_r(basis, ρ)
end


@doc raw"""
Get the lengthscale of the valence density for an atom with `n_elec_core` core
and `n_elec_valence` valence electrons.
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
