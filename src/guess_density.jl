@doc raw"""
    guess_density(basis)

Build a superposition of atomic densities (SAD) guess density.

We take for the guess density a gaussian centered around the atom, of
length specified by `atom_decay_length`, normalized to get the right number of electrons
```math
\hat{ρ}(G) = Z \exp\left(-(2π \text{length} |G|)^2\right)
```
"""
guess_density(basis::PlaneWaveBasis) = guess_density(basis, basis.model.atoms)
@timing function guess_density(basis::PlaneWaveBasis{T}, atoms) where {T}
    model = basis.model
    ρ = zeros(complex(T), basis.fft_size)
    # If no atoms, start with a zero initial guess
    isempty(atoms) && return from_fourier(basis, ρ; assume_real=true)
    # fill ρ with the (unnormalized) Fourier transform, ie ∫ e^{-iGx} ρ(x) dx
    for (spec, positions) in atoms
        n_el_val = n_elec_valence(spec)
        decay_length::T = atom_decay_length(spec)
        for (iG, G) in enumerate(G_vectors(basis))
            Gsq = sum(abs2, model.recip_lattice * G)
            form_factor::T = n_el_val * exp(-Gsq * decay_length^2)
            for r in positions
                ρ[iG] += form_factor * cis(-2T(π) * dot(G, r))
            end
        end
    end

    # projection in the normalized plane wave basis
    from_fourier(basis, ρ / sqrt(model.unit_cell_volume); assume_real=true)
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
