@testitem "DFTD3" tags=[:dont_test_mpi] begin
    using DFTK
    using SimpleDFTD3
    using ForwardDiff
    using LinearAlgebra

    # A triclinic cell (not a symmetric matrix) with four elements, such that mixing up the
    # lattice vectors or the atoms changes the result
    lattice   = [7.9 1.1 0.6; 0.3 8.4 1.3; -0.4 0.9 9.1]
    atoms     = ElementCoulomb.([:C, :H, :O, :Si, :H])
    positions = [[0.02, 0.05, 0.01], [0.21, 0.13, 0.17], [0.48, 0.41, 0.33],
                 [0.72, 0.66, 0.59], [0.35, 0.81, 0.77]]

    function d3_basis(d3; lattice=lattice, positions=positions)
        model = Model(lattice, atoms, positions; terms=[d3], symmetries=false)
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
        (; basis, term=only(basis.terms))
    end

    @testset "DFTD3 against reference values" begin
        # D3(BJ) for PBE from the Python package dftd3 1.6.0 (Cartesian gradient per atom)
        ref_energy   = -0.01969554988298249
        ref_gradient = [[0.0004196174532299252, 0.0003040636883882062, 0.0003444801691594112],
                        [-0.000504573474991558, -0.0002972659368946294, -0.0004414068334081278],
                        [0.0009405026265392106, 0.000955305382646449, 0.0010027667840649106],
                        [-0.0032740227643299944, 0.00031993157463630693, 0.0008490558133719237],
                        [0.0024184761595524174, -0.0012820347087763328, -0.0017548959331881177]]

        let (; basis, term) = d3_basis(DFTD3("pbe"))
            @test DFTK.energy(term, basis, nothing, nothing) ≈ ref_energy atol=1e-12
            forces = compute_forces(term, basis, nothing, nothing)
            @test stack(forces) ≈ -lattice' * stack(ref_gradient) atol=1e-12
        end

        # Test different versions of constructing the d3 model
        dispersion_model = DispersionModel(lattice, atoms, positions)
        for d3 in (DFTD3(PBE()), DFTD3(RationalDamping("pbe")), DFTD3("pbe", dispersion_model),
                   DFTD3(RationalDamping("pbe"), dispersion_model))
            let (; term, basis) = d3_basis(d3)
                @test DFTK.energy(term, basis, nothing, nothing) ≈ ref_energy atol=1e-12
            end
        end
    end

    @testset "DFTD3 three-body term and printing" begin
        # Damping, three-body term and switching widths against SimpleDFTD3 directly
        dispersion_model = DispersionModel(lattice, atoms, positions)
        set_realspace_cutoff!(dispersion_model; width2=5.0, width3=5.0)
        energy = dispersion(dispersion_model, ZeroDamping("pbe"; atm=true)).energy
        d3 = DFTD3("pbe"; damping="D3(0)", three_body=true,
                   switching_width_2body=5.0, switching_width_3body=5.0)
        (; term, basis) = d3_basis(d3)
        @test DFTK.energy(term, basis, nothing, nothing) ≈ energy atol=1e-12

        @test sprint(show, DFTD3("pbe")) ==
              "DFTD3(\"pbe\"; damping=\"D3(BJ)\", three_body=false)"
        for (functionals, method) in ((PBE0(), "pbe0"), ([:gga_x_pbe, :gga_c_pbe], "pbe"))
            @test sprint(show, DFTD3(functionals)) ==
                  "DFTD3(\"$method\"; damping=\"D3(BJ)\", three_body=false)"
        end
        @test sprint(show, d3) == "DFTD3(\"pbe\"; damping=\"D3(0)\", three_body=true, " *
                                  "switching_width_2body=5.0, switching_width_3body=5.0)"
        @test sprint(show, DFTD3(ZeroDamping("pbe"))) == "DFTD3(custom)"
        @test sprint(show, DFTD3("pbe", dispersion_model)) == "DFTD3(custom)"
    end

    @testset "DFTD3 errors" begin
        # Unknown damping identifier
        @test_throws ArgumentError d3_basis(DFTD3("pbe"; damping="D3(XYZ)"))
        # No D3 data beyond Z = 103
        model = Model(Matrix(8.0I, 3, 3), [ElementCoulomb(104)], [zeros(3)];
                      terms=[DFTD3("pbe")])
        @test_throws ArgumentError PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
        # Only fully periodic 3D systems
        lattice_2d = [8.0 1.0 0.0; 0.0 8.0 0.0; 0.0 0.0 0.0]
        model = Model(lattice_2d, [ElementCoulomb(:C)], [zeros(3)]; terms=[DFTD3("pbe")])
        @test_throws ArgumentError PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))

        # Phonons and derivatives of the forces are not implemented
        (; basis, term) = d3_basis(DFTD3("pbe"))
        @test_throws ErrorException DFTK.compute_dynmat(term, basis, nothing, nothing)

        T = typeof(ForwardDiff.Dual(0.0, 1.0))
        (; basis, term) = d3_basis(DFTD3("pbe"); lattice=Matrix{T}(lattice))
        @test_throws ErrorException compute_forces(term, basis, nothing, nothing)
    end
end

@testitem "DFTD3 derivatives" tags=[:dont_test_mpi] begin
    using DFTK
    using SimpleDFTD3
    using ForwardDiff
    using LinearAlgebra

    # A triclinic cell (not a symmetric matrix) with four elements, such that mixing up the
    # lattice vectors or the atoms changes the result
    lattice   = [7.9 1.1 0.6; 0.3 8.4 1.3; -0.4 0.9 9.1]
    atoms     = ElementCoulomb.([:C, :H, :O, :Si, :H])
    positions = [[0.02, 0.05, 0.01], [0.21, 0.13, 0.17], [0.48, 0.41, 0.33],
                 [0.72, 0.66, 0.59], [0.35, 0.81, 0.77]]

    # Atoms crossing a hard cutoff make the energy jump, which spoils finite differences.
    # The two-body and three-body terms are switched off smoothly. Every atom inside the
    # `cn` cutoff adds at least 1/(1+e^16) ≈ 1e-7 to the coordination number, so a small
    # `cn` cutoff puts fewer atoms near it. A custom dispersion model is used as is, so it
    # needs to be rebuilt for each structure.
    function smooth_d3_basis(; lattice=lattice, positions=positions)
        dispersion_model = DispersionModel(lattice, atoms, positions)
        set_realspace_cutoff!(dispersion_model; cn=15.0, width2=10.0, width3=10.0)
        d3 = DFTD3("b3lyp", dispersion_model; three_body=true)

        model = Model(lattice, atoms, positions; terms=[d3], symmetries=false)
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
        (; basis, term=only(basis.terms))
    end

    h = 1e-4

    function f(ε)
        strain = [0.0 0.3 0.1; 0.3 0.2 0.0; 0.1 0.0 -0.4]
        (; term, basis) = smooth_d3_basis(; lattice=(I + ε * strain) * lattice)
        DFTK.energy(term, basis, nothing, nothing)
    end
    @test ForwardDiff.derivative(f, 0.0) ≈ (f(h) - f(-h)) / 2h atol=1e-8

    displacement = [[0.1, -0.3, 0.2], [0.0, 0.4, -0.1], [-0.2, 0.1, 0.3],
                    [0.3, 0.0, -0.2], [0.1, 0.2, 0.0]]
    function g(ε::T) where {T}
        (; term, basis) = smooth_d3_basis(; positions=positions + ε * displacement,
                                            lattice=Matrix{T}(lattice))
        DFTK.energy(term, basis, nothing, nothing)
    end
    (; basis, term) = smooth_d3_basis()
    forces = compute_forces(term, basis, nothing, nothing)
    @test ForwardDiff.derivative(g, 0.0) ≈ -dot(forces, displacement) atol=1e-12
    @test ForwardDiff.derivative(g, 0.0) ≈ (g(h) - g(-h)) / 2h atol=1e-8
end

@testitem "DFTD3 in an SCF" tags=[:forces] begin
    using DFTK
    using SimpleDFTD3
    using LinearAlgebra
    using PseudoPotentialData

    # The D3 contribution to energy, forces and stresses, compared to SimpleDFTD3 directly
    lattice   = Matrix(6.0I, 3, 3)
    atoms     = fill(ElementPsp(:H, PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")), 2)
    positions = [[0.0, 0.0, 0.0], [0.12, 0.05, 0.03]]
    function run_scf(extra_terms)
        model = model_DFT(lattice, atoms, positions; functionals=LDA(), extra_terms)
        basis = PlaneWaveBasis(model; Ecut=8, kgrid=(1, 1, 1))
        self_consistent_field(basis; tol=1e-10)
    end
    damping    = RationalDamping("b3lyp")
    scfres_ref = run_scf([])
    scfres_d3  = run_scf([DFTD3(damping)])

    dispersion_model = DispersionModel([1, 1], lattice * stack(positions); lattice)
    (; energy, gradient, sigma) = dispersion(dispersion_model, damping)
    @test scfres_d3.energies["DFTD3"] ≈ energy atol=1e-12
    @test scfres_d3.energies.total - scfres_ref.energies.total ≈ energy atol=1e-8
    forces = compute_forces_cart(scfres_d3) - compute_forces_cart(scfres_ref)
    @test stack(forces) ≈ -gradient atol=1e-6
    stresses = compute_stresses_cart(scfres_d3) - compute_stresses_cart(scfres_ref)
    @test stresses ≈ sigma / det(lattice) atol=1e-6
end
