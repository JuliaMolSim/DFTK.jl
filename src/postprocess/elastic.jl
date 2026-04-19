import DifferentiationInterface as DI


function _stress_from_strain(basis0::PlaneWaveBasis, voigt_strain;
                             symmetries=true, kwargs_scf...)
    # TODO restart SCF from previous
    model0 = basis0.model
    lattice = DFTK.voigt_strain_to_full(voigt_strain) * model0.lattice
    model = Model(model0; lattice, symmetries)
    basis = PlaneWaveBasis(basis0; model)
    scfres = self_consistent_field(basis; kwargs_scf...)
    DFTK.full_stress_to_voigt(compute_stresses_cart(scfres))
end

"""
    elastic_tensor(scfres; magnetic_moments=[],
                   tol_symmetry=SYMMETRY_TOLERANCE,
                   kwargs_scf...)

Computes the *clamped-ion* elastic tensor (without ionic relaxation) via
automatic differentiation of the stress tensor with respect to strain.
Returns a named tuple `(; voigt_stress, C)` where `C[i,j] = ∂σᵢ/∂ηⱼ` is
the 6×6 elastic tensor in Voigt notation.

`tol_symmetry` controls the tolerance for symmetry detection on the
strained lattice. Extra keyword arguments are forwarded to
`self_consistent_field` (e.g. `tol`, `response`, `mixing`).

For cubic systems the three independent constants (C11, C12, C44) are
obtained from a single directional derivative; for other symmetries the
full Jacobian is computed.
"""
function elastic_tensor(scfres::NamedTuple;
                        magnetic_moments=[],
                        tol_symmetry=SYMMETRY_TOLERANCE,
                        tol=scfres.history_Δρ[end],
                        kwargs_scf...)
    basis0 = scfres.basis
    T = eltype(basis0)
    model0 = basis0.model
    η0 = zeros(T, 6)

    spg = Spglib.get_dataset(spglib_cell(model0, magnetic_moments))
    is_cubic = spg.pointgroup_symbol in ("23", "m-3", "432", "-43m", "m-3m")

    if is_cubic
        @assert spg.std_rotation_matrix == I(3) "Cubic symmetry optimization " *
                                                 "only works for non-rotated cells"
        strain_pattern = [1., 0., 0., 1., 0., 0.];  # recovers [C11, C12, C12, C44, 0, 0]

        # The finitely strained lattice is only used for symmetry determination
        displacement = 100 * tol_symmetry
        strained_lattice = DFTK.voigt_strain_to_full(
            displacement * strain_pattern) * model0.lattice
        symmetries_strain = symmetry_operations(strained_lattice,
                                                 model0.atoms, model0.positions;
                                                 tol_symmetry)

        # TODO unfold scfres partially to symmetries_strain and initialize 2nd scf with it

        stress_fn(η) = _stress_from_strain(basis0, η;
                                           symmetries=symmetries_strain,
                                           tol, kwargs_scf...)
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
        f(η) = _stress_from_strain(basis0, η; symmetries=false, tol, kwargs_scf...)
        (voigt_stress, C) = DI.value_and_jacobian(f, DI.AutoForwardDiff(), η0)
    end

    (; voigt_stress, C)
end
