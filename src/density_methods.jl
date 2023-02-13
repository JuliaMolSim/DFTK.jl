# abstract type DensityMethod                  end
# abstract type AtomicDensity <: DensityMethod end

# struct RandomDensity           <: DensityMethod end
# struct CoreDensity             <: AtomicDensity end
# struct ValenceGaussianDensity  <: AtomicDensity end
# struct ValenceNumericalDensity <: AtomicDensity end
# struct ValenceAutoDensity      <: AtomicDensity end

# # Random density method
# function guess_density(basis::PlaneWaveBasis, ::RandomDensity;
#                        n_electrons=basis.model.n_electrons)
#     random_density(basis, n_electrons)
# end

# function random_density(basis::PlaneWaveBasis, n_electrons)
#     ρtot  = rand(T, basis.fft_size)
#     ρtot  = ρtot .* n_electrons ./ (sum(ρtot) * basis.dvol)  # Integration to n_electrons
#     ρspin = nothing
#     if basis.model.n_spin_components > 1
#         ρspin = rand((-1, 1), basis.fft_size) .* rand(T, basis.fft_size) .* ρtot
#         @assert all(abs.(ρspin) .≤ ρtot)
#     end
#     ρ_from_total_and_spin(ρtot, ρspin)
# end

# # Atomic density methods
# function guess_density(basis::PlaneWaveBasis, magnetic_moments=[],
#                        n_electrons=basis.model.n_electrons)
#     atomic_density(basis, ValenceAutoDensity(), magnetic_moments, n_electrons)
# end

# function guess_density(basis::PlaneWaveBasis, method::AtomicDensity, magnetic_moments=[];
#                        n_electrons=basis.model.n_electrons)
#     atomic_density(basis, method, magnetic_moments, n_electrons)
# end

# function atomic_density(basis::PlaneWaveBasis, method::AtomicDensity, magnetic_moments,
#                         n_electrons)
#     ρtot = atomic_total_density(basis, method)
#     ρspin = atomic_spin_density(basis, method, magnetic_moments)
#     ρ = ρ_from_total_and_spin(ρtot, ρspin)
    
#     N = sum(ρ) * basis.model.unit_cell_volume / prod(basis.fft_size)

#     if !isnothing(n_electrons) && (N > 0)
#         ρ .*= n_electrons / N  # Renormalize to the correct number of electrons
#     end
#     ρ
# end

# function atomic_total_density(basis::PlaneWaveBasis, method::AtomicDensity; coefficients)
#     form_factors = atomic_density_form_factors(basis, method)
#     atomic_density_superposition(basis, form_factors; coefficients)
# end

# function atomic_spin_density(basis::PlaneWaveBasis, method::AtomicDensity, magnetic_moments)
#     model = basis.model
#     if model.spin_polarization in (:none, :spinless)
#         isempty(magnetic_moments) && return nothing
#         error("Initial magnetic moments can only be used with collinear models.")
#     end

#     # If no magnetic moments start with a zero spin density
#     magmoms = Vec3{T}[normalize_magnetic_moment(magmom) for magmom in magnetic_moments]
#     if all(iszero, magmoms)
#         @warn("Returning zero spin density guess, because no initial magnetization has " *
#               "been specified in any of the given elements / atoms. Your SCF will likely " *
#               "not converge to a spin-broken solution.")
#         return zeros(T, basis.fft_size)
#     end

#     @assert length(magmoms) == length(basis.model.atoms)
#     coefficients = map(zip(basis.model.atoms, magmoms)) do (atom, magmom)
#         iszero(magmom[1:2]) || error("Non-collinear magnetization not yet implemented")
#         magmom[3] ≤ n_elec_valence(atom) || error(
#             "Magnetic moment $(magmom[3]) too large for element $(atomic_symbol(atom)) " *
#             "with only $(n_elec_valence(atom)) valence electrons."
#         )
#         magmom[3] / n_elec_valence(atom)
#     end

#     form_factors = atomic_density_form_factors(basis, method)
#     atomic_density_superposition(basis, form_factors; coefficients)    
# end

# function atomic_density_superposition(basis::PlaneWaveBasis{T},
#                                       form_factors::IdDict{Tuple{Int,T},T};
#                                       coefficients=ones(T, length(basis.model.atoms))
#                                       )::Array{T,3} where {T}
#     model = basis.model
#     G_cart = G_vectors_cart(basis)

#     ρ = map(enumerate(G_vectors(basis))) do (iG, G)
#         Gnorm = norm(G_cart[iG])
#         ρ_iG = sum(enumerate(model.atom_groups); init=zero(Complex{T})) do (igroup, group)
#             sum(group) do iatom
#                 structure_factor::Complex{T} = cis2pi(-dot(G, model.positions[iatom]))
#                 coefficients[iatom]::T * form_factors[(igroup, Gnorm)]::T * structure_factor
#             end
#         end
#         ρ_iG / sqrt(model.unit_cell_volume)
#     end
#     enforce_real!(basis, ρ)  # Symmetrize Fourier coeffs to have real iFFT
#     irfft(basis, ρ)
# end

# function atomic_density_form_factors(basis::PlaneWaveBasis{T},
#                                      ::AtomicDensity)::IdDict{Tuple{Int,T},T} where {T<:Real}
#     model = basis.model
#     form_factors = IdDict{Tuple{Int,T},T}()  # IdDict for Dual compatability
#     for G in G_vectors_cart(basis)
#         Gnorm = norm(G)
#         for (igroup, group) in enumerate(model.atom_groups)
#             if !haskey(form_factors, (igroup, Gnorm))
#                 element = model.atoms[first(group)]
#                 form_factor = atomic_density(element.psp, Gnorm)
#                 form_factors[(igroup, Gnorm)] = form_factor
#             end
#         end
#     end
#     form_factors
# end

# function atomic_density(element::Element, Gnorm::T,
#                         ::ValenceGaussianDensity)::T where {T <: Real}
#     gaussian_valence_charge_density_fourier(element, Gnorm)
# end

# function atomic_density(element::Element, Gnorm::T,
#                         ::ValenceNumericalDensity)::T where {T <: Real}
#     eval_psp_density_valence_fourier(element.psp, Gnorm)
# end

# function atomic_density(element::Element, Gnorm::T,
#                         ::ValenceAutoDensity)::T where {T <: Real}
#     valence_charge_density_fourier(element, Gnorm)
# end

# function atomic_density(element::Element, Gnorm::T,
#                         ::CoreDensity)::T where {T <: Real}
#     if has_density_core(element)
#         core_charge_density_fourier(element, Gnorm)
#     else
#         zero(T)
#     end
# end
