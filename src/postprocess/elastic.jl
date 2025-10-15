import DifferentiationInterface as DI


function symmetries_from_strain(model0, voigt_strain)
    lattice = DFTK.voigt_strain_to_full(voigt_strain) * model0.lattice
    model = Model(model0; lattice, symmetries=true)
    model.symmetries
end

function stress_from_strain(model0, voigt_strain; symmetries, Ecut, kgrid, tol, kwargs...)
    # TODO restart SCF from previous
    lattice = DFTK.voigt_strain_to_full(voigt_strain) * model0.lattice
    model = Model(model0; lattice, symmetries)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; tol, kwargs...)
    DFTK.full_stress_to_voigt(compute_stresses_cart(scfres))
end

"""Computes the *clamped-ion* elastic constants (without ionic relaxation)"""
function elastic_constants(scfres::NamedTuple; 
                           magnetic_moments=[],
                           Ecut=scfres.basis.Ecut,
                           kgrid=scfres.basis.kgrid,
                           tol=scfres.history_Δρ[end],
                           kwargs...)    
    T = eltype(scfres.basis)
    model0 = scfres.basis.model
    η0 = zeros(T, 6)

    spg = Spglib.get_dataset(spglib_cell(model0, magnetic_moments))
    is_cubic = spg.pointgroup_symbol in ("23", "m-3", "432", "-43m", "m-3m")

    if is_cubic
        @assert spg.std_rotation_matrix == I(3) "Cubic symmetry optimization only works for non-rotated cells"
        strain_pattern = [1., 0., 0., 1., 0., 0.];  # recovers [C11, C12, C12, C44, 0, 0]
        symmetries_strain = symmetries_from_strain(model0, 0.01 * strain_pattern)

        # TODO unfold scfres partially to symmetries_strain and initialize 2nd scf with it

        stress_fn(η) = stress_from_strain(model0, η; symmetries=symmetries_strain,
                                          Ecut, kgrid, tol)
        voigt_stress, (dstress,) = DI.value_and_pushforward(stress_fn, DI.AutoForwardDiff(),
                                                            η0, (strain_pattern,))
        (C11, C12, _, C44, _, _) = dstress
        C = [C11 C12 C12 0   0   0;
             C12 C11 C12 0   0   0;
             C12 C12 C11 0   0   0;
             0   0   0   C44 0   0;
             0   0   0   0   C44 0;
             0   0   0   0   0   C44]
    # TODO add hexagonal, tetragonal, etc. cases here
    else
        # General elastic constants fallback: no symmetries & 6 perturbation
        f(η) = stress_from_strain(model0, η; symmetries=false, Ecut, kgrid, tol)
        (voigt_stress, C) = DI.value_and_jacobian(f, DI.AutoForwardDiff(), η0)
    end

    (; voigt_stress, C)
end
