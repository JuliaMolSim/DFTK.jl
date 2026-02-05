import DifferentiationInterface as DI


function _stress_from_strain(basis0::PlaneWaveBasis, voigt_strain; symmetries=true, tol)
    # TODO restart SCF from previous
    model0 = basis0.model
    lattice = DFTK.voigt_strain_to_full(voigt_strain) * model0.lattice
    new_model = Model(model0; lattice, symmetries)
    new_basis = PlaneWaveBasis(new_model,
                               basis0.Ecut, basis0.fft_size, basis0.variational,
                               basis0.kgrid, basis0.symmetries_respect_rgrid,
                               basis0.use_symmetries_for_kpoint_reduction,
                               basis0.comm_kpts, basis0.architecture)
    scfres = self_consistent_field(new_basis; tol)
    DFTK.full_stress_to_voigt(compute_stresses_cart(scfres))
end

"""
Computes the *clamped-ion* elastic tensor (without ionic relaxation).
Returns the elastic tensor in Voigt notation as the matrix C[i,j] = ∂σᵢ/∂ηⱼ.
"""
function elastic_tensor(scfres::NamedTuple;
                        magnetic_moments=[],
                        tol=scfres.history_Δρ[end])
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
        strained_lattice = DFTK.voigt_strain_to_full(0.01 * strain_pattern) * model0.lattice
        symmetries_strain = symmetry_operations(strained_lattice,
                                                 model0.atoms, model0.positions)

        # TODO unfold scfres partially to symmetries_strain and initialize 2nd scf with it

        stress_fn(η) = _stress_from_strain(basis0, η; symmetries=symmetries_strain, tol)
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
        f(η) = _stress_from_strain(basis0, η; symmetries=false, tol)
        (voigt_stress, C) = DI.value_and_jacobian(f, DI.AutoForwardDiff(), η0)
    end

    (; voigt_stress, C)
end
