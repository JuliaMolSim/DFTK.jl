import DifferentiationInterface as DI


function _stress_from_strain(basis0::PlaneWaveBasis, voigt_strain;
                             symmetries=true, ρ, kwargs_scf...)
    model0 = basis0.model
    lattice = DFTK.voigt_strain_to_full(voigt_strain) * model0.lattice
    model = Model(model0; lattice, symmetries)
    basis = PlaneWaveBasis(basis0; model)
    scfres = self_consistent_field(basis; ρ, kwargs_scf...)
    DFTK.full_stress_to_voigt(compute_stresses_cart(scfres))
end

"""
    elastic_tensor(scfres;
                   response=ResponseOptions(),
                   tol_symmetry=SYMMETRY_TOLERANCE)

Computes the *clamped-ion* elastic tensor (without ionic relaxation) via
automatic differentiation of the stress tensor with respect to strain.
Returns a named tuple `(; voigt_stress, C)` where `C[i,j] = ∂σᵢ/∂ηⱼ` is
the 6×6 elastic tensor in Voigt notation.

`response` controls the implicit response solver
(`solve_ΩplusK_split`) performed when the SCF is differentiated.

`tol_symmetry` controls the tolerance for symmetry detection on the
strained lattice.

For cubic systems the three independent constants (C11, C12, C44) are
obtained from a single directional derivative; for other symmetries the
full Jacobian is computed.
"""
function elastic_tensor(scfres::NamedTuple;
                        response=ResponseOptions(),
                        tol_symmetry=SYMMETRY_TOLERANCE,
                        magnetic_moments=[])  # TODO remove magnetic_moments after #1307
    # TODO factor this out into a `kwargs_scf_inherit(scfres)` helper once its
    # shape has settled (see also the analogous `kwargs_scf_checkpoints`).
    # Since scfres is converged, we tighten `diagtol_first` so the first
    # diagonalization in a warm-started strained SCF is not unnecessarily loose.
    diagtolalg = scfres.diagtolalg
    diagtol_first = determine_diagtol(diagtolalg, scfres)
    diagtolalg = AdaptiveDiagtol(; diagtol_first,
                                   diagtolalg.diagtol_max,
                                   diagtolalg.diagtol_min,
                                   diagtolalg.ratio_ρdiff)
    kwargs_scf = (; scfres.is_converged,
                    scfres.mixing,
                    damping=scfres.α,
                    scfres.nbandsalg,
                    scfres.fermialg,
                    diagtolalg,
                    scfres.solver,
                    scfres.eigensolver)
    basis0 = scfres.basis
    T = eltype(basis0)
    model0 = basis0.model
    η0 = zeros(T, 6)

    spg = Spglib.get_dataset(spglib_cell(model0, magnetic_moments))
    is_cubic = spg.pointgroup_symbol in ("23", "m-3", "432", "-43m", "m-3m")

    if is_cubic
        @assert spg.std_rotation_matrix == I(3) "Cubic symmetry optimization " *
                                                 "only implemented for non-rotated cells"
        strain_pattern = [1., 0., 0., 1., 0., 0.];  # recovers [C11, C12, C12, C44, 0, 0]

        # The finitely strained lattice is only used for symmetry determination
        displacement = 100 * tol_symmetry
        strained_lattice = DFTK.voigt_strain_to_full(
            displacement * strain_pattern) * model0.lattice
        symmetries_strain = symmetry_operations(strained_lattice,
                                                model0.atoms, model0.positions;
                                                tol_symmetry)

        stress_fn(η) = _stress_from_strain(basis0, η;
                                           symmetries=symmetries_strain,
                                           ρ=scfres.ρ,
                                           response, kwargs_scf...)
        voigt_stress, (dstress,) = DI.value_and_pushforward(
            stress_fn, DI.AutoForwardDiff(), η0, (strain_pattern,))
        (C11, C12, _, C44, _, _) = dstress
        C = [C11 C12 C12 0   0   0;
             C12 C11 C12 0   0   0;
             C12 C12 C11 0   0   0;
             0   0   0   C44 0   0;
             0   0   0   0   C44 0;
             0   0   0   0   0   C44]
    # TODO add hexagonal, tetragonal, etc. cases here
    else
        # General elastic constants fallback: no symmetries & 6 strain perturbations
        f(η) = _stress_from_strain(basis0, η; symmetries=false,
                                   ρ=scfres.ρ, response, kwargs_scf...)
        (voigt_stress, C) = DI.value_and_jacobian(f, DI.AutoForwardDiff(), η0)
    end

    (; voigt_stress, C)
end

struct ElasticPerturbations <: PerturbationSet end
Base.length(::ElasticPerturbations) = 6

function symmetrize_δρs(::ElasticPerturbations, basis::PlaneWaveBasis{T}, δρs) where {T}
    @assert length(δρs) == 6
    map(1:6) do iη
        δρ_sym = zeros_like(δρs[iη])
        strain = zeros(T, 6)
        strain[iη] = one(T)
        ϵ = voigt_strain_to_full(strain)
        for symop in basis.symmetries
            W_cart = matrix_red_to_cart(basis.model, symop.W)
            ϵ_transformed = W_cart * ϵ / W_cart
            strain_transformed = full_strain_to_voigt(ϵ_transformed)
            for j in 1:6
                if abs(strain_transformed[j]) > sqrt(eps(T))
                    δρ_sym .+= strain_transformed[j] .* apply_symop(symop, basis, δρs[j])
                end
            end
        end
        δρ_sym / length(basis.symmetries)
    end
end

"""
    elastic_tensor(scfres;
                   response=ResponseOptions(),
                   tol_symmetry=SYMMETRY_TOLERANCE)

Computes the *clamped-ion* elastic tensor (without ionic relaxation) via
automatic differentiation of the stress tensor with respect to strain.
Returns a named tuple `(; voigt_stress, C)` where `C[i,j] = ∂σᵢ/∂ηⱼ` is
the 6×6 elastic tensor in Voigt notation.

`response` controls the implicit response solver
(`solve_ΩplusK_split`) performed when the SCF is differentiated.

`tol_symmetry` controls the tolerance for symmetry detection on the
strained lattice.

For cubic systems the three independent constants (C11, C12, C44) are
obtained from a single directional derivative; for other symmetries the
full Jacobian is computed.
"""
function elastic_tensor_v2(scfres::NamedTuple;
                           response=ResponseOptions())
    basis0 = scfres.basis
    T = eltype(basis0)
    model0 = basis0.model

    function make_strained_basis(η)
        lattice = voigt_strain_to_full(η) * model0.lattice
        # explicitly keep model0's symmetries!
        model = Model(model0; lattice, symmetries=model0.symmetries)
        PlaneWaveBasis(basis0; model)
    end

    Tag = typeof(ForwardDiff.Tag(make_strained_basis, T))
    ε = Dual{Tag}(zero(T), one(T))

    δHψs = map(1:6) do istrain
        δη = zeros(T, 6)
        δη[istrain] = one(T)
        basis = make_strained_basis(ε .* δη)
        # TODO: τ, occupation_threshold
        ρ = compute_density(basis, scfres.ψ, scfres.occupation)
        # τ = isnothing(scfres.τ) ? nothing : compute_kinetic_energy_density(strained_basis, scfres.ψ, scfres.occupation)
        ham = energy_hamiltonian(basis, scfres.ψ, scfres.occupation;
                                 ρ, scfres.eigenvalues, scfres.εF).ham
        ForwardDiff.extract_derivative(Tag, ham * scfres.ψ)
    end

    tol = last(scfres.history_Δρ)
    dfpt_res = solve_ΩplusK_split(scfres, δHψs, ElasticPerturbations(); tol)

    symmetries_id_only = [one(first(basis0.symmetries))]
    δρs = map(1:6) do istrain
        δη = zeros(T, 6)
        δη[istrain] = one(T)
        # TODO: building a full basis just to compute ρ is rather wasteful
        basis = make_strained_basis(ε .* δη)
        ψ = scfres.ψ .+ ε .* dfpt_res.δψs[istrain]
        occupation = scfres.occupation .+ ε .* dfpt_res.δoccupations[istrain]
        ForwardDiff.extract_derivative(Tag, compute_density(basis, ψ, occupation; symmetries=symmetries_id_only))
    end
    δρs = symmetrize_δρs(ElasticPerturbations(), basis0, δρs)

    C_cols = map(1:6) do istrain
        δη = zeros(T, 6)
        δη[istrain] = one(T)
        basis = make_strained_basis(ε * δη)
        ψ = scfres.ψ .+ ε .* dfpt_res.δψs[istrain]
        occupation = scfres.occupation .+ ε .* dfpt_res.δoccupations[istrain]
        eigenvalues = scfres.eigenvalues .+ ε .* dfpt_res.δeigenvaluess[istrain]
        ρ = scfres.ρ .+ ε .* δρs[istrain]
        εF = scfres.εF + ε * dfpt_res.δεFs[istrain]
        # TODO: τ = isnothing(scfres.τ) ? nothing : compute_kinetic_energy_density(strained_basis, ψ, scfres.occupation)
        σ = compute_stresses_cart(basis, ψ, occupation; eigenvalues, εF, symmetries=symmetries_id_only, ρ)
        ForwardDiff.extract_derivative(Tag, full_stress_to_voigt(σ))
    end

    C = symmetrize_elastic_tensor(model0, stack(C_cols); basis0.symmetries)
    (; C, dfpt_res)
end
