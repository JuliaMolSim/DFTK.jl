# Compute densities of states

# IDOS (integrated density of states)
# N(ε) = sum_n f_n = sum_n f((εn-ε)/temperature)
# DOS (density of states)
# D(ε) = N'(ε)
#
# LDOS (local density of states)
# LD(ε) = sum_n f_n' |ψn|^2 = sum_n δ(ε - ε_n) |ψn|^2

"""
Total density of states at energy ε
"""
function compute_dos(ε, basis, eigenvalues; smearing=basis.model.smearing,
                     temperature=basis.model.temperature)
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_dos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    D = zeros(typeof(ε), basis.model.n_spin_components)
    for σ = 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        for (iband, εnk) in enumerate(eigenvalues[ik])
            enred = (εnk - ε) / temperature
            D[σ] -= (filled_occ * basis.kweights[ik] / temperature
                     * Smearing.occupation_derivative(smearing, enred))
        end
    end
    D = mpi_sum(D, basis.comm_kpts)
end
function compute_dos(scfres::NamedTuple; ε=scfres.εF, kwargs...)
        compute_dos(ε, scfres.basis, scfres.eigenvalues; kwargs...)
end

"""
Local density of states, in real space. `weight_threshold` is a threshold
to screen away small contributions to the LDOS.
"""
function compute_ldos(ε, basis::PlaneWaveBasis{T}, eigenvalues, ψ;
                      smearing=basis.model.smearing,
                      temperature=basis.model.temperature,
                      weight_threshold=eps(T)) where {T}
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_ldos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    weights = deepcopy(eigenvalues)
    for (ik, εk) in enumerate(eigenvalues)
        for (iband, εnk) in enumerate(εk)
            enred = (εnk - ε) / temperature
            weights[ik][iband] = (-filled_occ / temperature
                                  * Smearing.occupation_derivative(smearing, enred))
        end
    end

    # Use compute_density routine to compute LDOS, using just the modified
    # weights (as "occupations") at each k-point. Note, that this automatically puts in the
    # required symmetrization with respect to kpoints and BZ symmetry
    compute_density(basis, ψ, weights; occupation_threshold=weight_threshold)
end
function compute_ldos(scfres::NamedTuple; ε=scfres.εF, kwargs...)
    compute_ldos(ε, scfres.basis, scfres.eigenvalues, scfres.ψ; kwargs...)
end

"""
Plot the density of states over a reasonable range. Requires to load `Plots.jl` beforehand.
"""
function plot_dos end

function plot_ldos end

"""
Projected density of states at energy ε
"""
# PD(ε) = sum_n f_n' |<ψn,ϕ>|^2
function compute_pdos_projs(basis, eigenvalues, ψ, gtest, position)


    projs = similar(eigenvalues)

    for (ik, ψk) in enumerate(ψ)
        Gik = DFTK.Gplusk_vectors_cart(basis, basis.kpoints[ik])
        pG = [dot(position, Gi) for Gi in Gik]
        gik = exp.(-im * pG) .* gtest.(Gik)
        #gik = gtest.(Gik)
        projs[ik] = abs2.(ψk' * gik)
    end

    projs ./ (basis.model.unit_cell_volume)
end

function compute_pdos(ε, basis, eigenvalues, projs;
    smearing=basis.model.smearing,
    temperature=basis.model.temperature)

    if (temperature == 0) || smearing isa Smearing.None
        error("compute_dos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    D = zeros(typeof(ε), basis.model.n_spin_components)
    for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        for (iband, εnk) in enumerate(eigenvalues[ik])
            enred = (εnk - ε) / temperature
            D[σ] -= (filled_occ * basis.kweights[ik] * projs[ik][iband] / temperature
                     *
                     Smearing.occupation_derivative(smearing, enred))
        end
    end
    D = mpi_sum(D, basis.comm_kpts)
end

function compute_pdos(ε, basis, eigenvalues, ψ, gtest, position; kwargs...)
    projs = compute_pdos_projs(basis, eigenvalues, ψ, gtest, position)
    map(x -> compute_pdos(x, basis, eigenvalues, projs; kwargs...), ε)
end

function compute_pdos(scfres::NamedTuple, gtset, position;
    ε=scfres.εF, kwargs...)
    compute_pdos(ε, scfres.basis, scfres.eigenvalues, scfres.ψ, gtset, position; kwargs...)
end

function atom_fourier(i::Integer, l::Integer, m::Integer, q::AbstractVector{T}, psp_upf) where {T<:Real}
    im^l * ylm_real(l, m, -q / (norm(q) + 1e-10)) * eval_psp_pswfc_fourier(psp_upf, i, l, norm(q))
end

function compute_pdos(ε, i::Integer, l::Integer, m::Integer, 
    iatom::Integer, basis, eigenvalues, ψ; kwargs...)
    compute_pdos(ε, basis, eigenvalues, ψ, q -> atom_fourier(i, l, m, q, basis.model.atoms[1].psp), basis.model.positions[iatom]; kwargs...)
end

function compute_pdos(i::Integer, l::Integer, m::Integer, iatom::Integer, scfres::NamedTuple;
    ε=scfres.εF, kwargs...)
    compute_pdos(ε, i, l, m, iatom, scfres.basis, scfres.eigenvalues, scfres.ψ; kwargs...)
end

function plot_pdos end
