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

    Tag = typeof(ForwardDiff.Tag(ElasticPerturbations(), T))
    εs = map(1:6) do istrain
        Dual{Tag}(zero(T), ntuple(i -> i == istrain ? one(T) : zero(T), Val(6))...)
    end
    lattice = voigt_strain_to_full(εs) * model0.lattice
    # explicitly keep model0's symmetries!
    model = Model(model0; lattice, symmetries=model0.symmetries)
    basis = PlaneWaveBasis(basis0; model)

    δHψs = let
        # TODO: τ, occupation_threshold
        ρ = compute_density(basis, scfres.ψ, scfres.occupation)
        ham = energy_hamiltonian(basis, scfres.ψ, scfres.occupation;
                                 ρ, scfres.eigenvalues, scfres.εF).ham
        δHψ = ham * scfres.ψ
        map(1:6) do istrain
            [ForwardDiff.partials.(Tag, δHψk, istrain) for δHψk in δHψ]
        end
    end

    tol = last(scfres.history_Δρ)
    dfpt_res = solve_ΩplusK_split(scfres, δHψs, ElasticPerturbations(); tol)

    # Careful that the resulting δρs do not account for basis_dual contribution to strain yet
    ψ = map(scfres.ψ, dfpt_res.δψs...) do ψk, δψk...
        map(ψk, δψk...) do ψnk, δψnk...
            Complex(Dual{Tag}(real(ψnk), real.(δψnk)),
                    Dual{Tag}(imag(ψnk), imag.(δψnk)))
        end
    end
    eigenvalues = map(scfres.eigenvalues, dfpt_res.δeigenvaluess...) do εk, δεk...
        map((εnk, δεnk...) -> Dual{Tag}(εnk, δεnk), εk, δεk...)
    end
    occupation = map(scfres.occupation, dfpt_res.δoccupations...) do occk, δocck...
        occk_cpu = to_cpu(occk)
        to_device(basis.architecture,
                  map((occk_cpu, δocck...) -> Dual{Tag}(occk_cpu, δocck), occk_cpu, δocck...))
    end
    εF = Dual{Tag}(scfres.εF, dfpt_res.δεFs...)

    sym_id = [one(first(basis.symmetries))]
    σ = full_stress_to_voigt(compute_stresses_cart(basis, ψ, occupation; eigenvalues, εF, symmetries=sym_id))
    C_nosym = [ForwardDiff.partials(Tag, σ[i], j) for i in 1:6, j in 1:6]
    C = symmetrize_elastic_tensor(model0, C_nosym; basis0.symmetries)

    (; C, dfpt_res)
end

const ElasticDual{V} = ForwardDiff.Dual{ForwardDiff.Tag{ElasticPerturbations,V},V,6}

# This is needed to differentiate through compute_stresses_cart
# Let ε1 be the outer dual (the one carrying the 6 elastic perturbations from DFPT)
# and ε2 be the inner dual (another elastic perturbation using Hellmann-Feynman)
# Overall the density ρ = ρ0 + ρ1 ε1 + ρ2 ε2 + ρ12 ε1 ε2
# ρ0 and ρ2 are symmetrized as usual since they are built from the unperturbed ψ,
# whereas ρ1 and ρ12 are built from a perturbed ψ and need to be symmetrized by mixing
# the perturbations.
# We will anyway symmetrize the elastic tensor, so as long as the elastic tensor is linear in the
# considered density component we can skip symmetrizing it.
# For a second order derivative we will have products of order 0 * order 2 and products of order 1 * order 1
# So we can get away with only symmetrizing ρ0 and ρ2 and leaving the others alone!
# With the usual nesting the representation is (ρ0 + ρ1 ε1) + (ρ2 + ρ12 ε1) ε2
# TODO: according to Claude we would still need to symmetrize ρ2 for meta-GGA (so τ) or USPP,
#       so we should instead symmetrize ρ1 for which all required partials are already available.
@views @timing function symmetrize_ρ(basis, ρ::AbstractArray{<:Dual{Tg,<:ElasticDual{V}}};
                                     symmetries=basis.symmetries) where {Tg, V<:AbstractFloat}
    # Due to the ffts inside symmetrize_̢ρ, we have to build a non-dual basis
    # To keep this reasonably cheap, we first build a non-dual model with no terms
    # and then construct a new basis.
    # Quite awkward but yeah... Maybe we could instead pass an fft_grid to symmetrize_ρ
    model0 = construct_value(construct_value(basis.model))
    model0 = Model(model0; terms=[Kinetic()])
    basis0 = PlaneWaveBasis(model0, ForwardDiff.value(ForwardDiff.value(basis.Ecut)),
                   basis.fft_size, basis.variational,
                   basis.kgrid, basis.symmetries_respect_rgrid,
                   basis.use_symmetries_for_kpoint_reduction,
                   basis.comm_kpts, basis.architecture)

    npartials_2 = ForwardDiff.npartials(eltype(ρ))
    ρ0_ρ1 = symmetrize_value_only(basis0, ForwardDiff.value.(Tg, ρ), symmetries)
    ρ2_ρ12 = map(1:npartials_2) do α
        symmetrize_value_only(basis0, ForwardDiff.partials.(Tg, ρ, α), symmetries)
    end
    map(ρ0_ρ1, ρ2_ρ12...) do v, dv...
        ForwardDiff.Dual{Tg}(v, dv...)
    end
end

function symmetrize_value_only(basis, ρ::AbstractArray{<:Dual{Tg}}, symmetries) where {Tg}
    ρ0 = ForwardDiff.value.(ρ)
    ρ0 = symmetrize_ρ(basis, ρ0; symmetries)
    map(ρ0, ρ) do ρ0i, ρi
        Dual{Tg}(ρ0i, ForwardDiff.partials(ρi))
    end
end
