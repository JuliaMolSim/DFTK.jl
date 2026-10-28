@testmodule DFTD3System begin
using DFTK
# A triclinic cell (not a symmetric matrix) with four elements, such that mixing up the
# lattice vectors or the atoms changes the result
lattice   = [7.9 1.1 0.6; 0.3 8.4 1.3; -0.4 0.9 9.1]
atoms     = ElementCoulomb.([:C, :H, :O, :Si, :H])
positions = [[0.02, 0.05, 0.01], [0.21, 0.13, 0.17], [0.48, 0.41, 0.33],
             [0.72, 0.66, 0.59], [0.35, 0.81, 0.77]]

function d3_term(lattice, positions, d3)
    model = Model(lattice, atoms, positions; terms=[d3], symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
    (; basis, term=only(basis.terms))
end
function d3_energy(lattice, positions, d3)
    (; basis, term) = d3_term(lattice, positions, d3)
    DFTK.energy(term, basis, nothing, nothing)
end
end


@testitem "DFTD3 against reference values" tags=[:dont_test_mpi] setup=[DFTD3System] begin
    using DFTK
    using SimpleDFTD3
    using .DFTD3System: lattice, atoms, positions, d3_term

    # D3(BJ) for PBE from the Python package dftd3 1.6.0 (Cartesian gradient per atom)
    ref_energy   = -0.01969554988298249
    ref_gradient = [[0.0004196174532299252, 0.0003040636883882062, 0.0003444801691594112],
                    [-0.000504573474991558, -0.0002972659368946294, -0.0004414068334081278],
                    [0.0009405026265392106, 0.000955305382646449, 0.0010027667840649106],
                    [-0.0032740227643299944, 0.00031993157463630693, 0.0008490558133719237],
                    [0.0024184761595524174, -0.0012820347087763328, -0.0017548959331881177]]

    (; basis, term) = d3_term(lattice, positions, DFTD3(RationalDamping("pbe")))
    @test DFTK.energy(term, basis, nothing, nothing) ≈ ref_energy atol=1e-12
    forces = compute_forces(term, basis, nothing, nothing)
    @test stack(forces) ≈ -lattice' * stack(ref_gradient) atol=1e-12

    # By default the damping is chosen from the functional of the model
    model = model_DFT(lattice, atoms, positions; functionals=PBE(), extra_terms=[DFTD3()])
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
    @test DFTK.energy(last(basis.terms), basis, nothing, nothing) ≈ ref_energy atol=1e-12

    @test sprint(show, DFTD3()) == "DFTD3()"
    @test sprint(show, DFTD3(ZeroDamping("pbe"); cutoffs=(; cn=30.0))) ==
          "DFTD3(D3(0); cutoffs=(cn = 30.0,))"
end

@testitem "DFTD3 derivatives" tags=[:dont_test_mpi] setup=[DFTD3System] begin
    using DFTK
    using SimpleDFTD3
    using ForwardDiff
    using LinearAlgebra
    using .DFTD3System: lattice, positions, d3_term, d3_energy

    # Atoms crossing a hard cutoff make the energy jump, which spoils finite differences.
    # Pair terms are tiny at a large `disp2` cutoff. Every atom inside the `cn` cutoff adds
    # at least 1/(1+e^16) ≈ 1e-7 to the coordination number, so a small `cn` cutoff puts
    # fewer atoms near it.
    d3 = DFTD3(RationalDamping("b3lyp"); cutoffs=(; disp2=150.0, cn=15.0))
    h = 1e-4

    # Strain, as used for the stresses
    strain = [0.0 0.3 0.1; 0.3 0.2 0.0; 0.1 0.0 -0.4]
    f(ε) = d3_energy((I + ε * strain) * lattice, positions, d3)
    @test ForwardDiff.derivative(f, 0.0) ≈ (f(h) - f(-h)) / 2h atol=2e-6

    # Moving the atoms, which needs the lattice to have the same number type
    displacement = [[0.1, -0.3, 0.2], [0.0, 0.4, -0.1], [-0.2, 0.1, 0.3],
                    [0.3, 0.0, -0.2], [0.1, 0.2, 0.0]]
    g(ε) = d3_energy(Matrix{typeof(ε)}(lattice), positions + ε * displacement, d3)
    (; basis, term) = d3_term(lattice, positions, d3)
    forces = compute_forces(term, basis, nothing, nothing)
    @test ForwardDiff.derivative(g, 0.0) ≈ -dot(forces, displacement) atol=1e-12
    @test ForwardDiff.derivative(g, 0.0) ≈ (g(h) - g(-h)) / 2h atol=1e-8
end

@testitem "DFTD3 errors" tags=[:dont_test_mpi] setup=[DFTD3System] begin
    using DFTK
    using SimpleDFTD3
    using ForwardDiff
    using .DFTD3System: lattice, atoms, positions, d3_term, d3_energy

    # No functional to choose the damping from
    @test_throws ArgumentError d3_term(lattice, positions, DFTD3())
    # The stored parameters only fit the unmodified functional
    function default_d3_basis(functionals)
        model = model_DFT(lattice, atoms, positions; functionals, extra_terms=[DFTD3()])
        PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
    end
    @test default_d3_basis(PBE0()) isa PlaneWaveBasis
    @test_throws ArgumentError default_d3_basis(PBE0(; exx_fraction=0.5))
    @test_throws ArgumentError default_d3_basis(Xc([:gga_x_pbe, :gga_c_pbe];
                                                    scaling_factor=0.5))
    # No D3 data beyond Z = 103
    model = Model(lattice, [ElementCoulomb(104)], [zeros(3)];
                  terms=[DFTD3(RationalDamping("pbe"))])
    @test_throws ArgumentError PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
    # Only fully periodic 3D systems
    lattice_2d = [8.0 1.0 0.0; 0.0 8.0 0.0; 0.0 0.0 0.0]
    model = Model(lattice_2d, [ElementCoulomb(:C)], [zeros(3)];
                  terms=[DFTD3(RationalDamping("pbe"))])
    @test_throws ArgumentError PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))

    # Phonons and derivatives of the forces are not implemented
    (; basis, term) = d3_term(lattice, positions, DFTD3(RationalDamping("pbe")))
    @test_throws ErrorException DFTK.compute_dynmat(term, basis, nothing, nothing)
    T = typeof(ForwardDiff.Dual(0.0, 1.0))
    (; basis, term) = d3_term(Matrix{T}(lattice), positions, DFTD3(RationalDamping("pbe")))
    @test_throws ErrorException compute_forces(term, basis, nothing, nothing)

    # Second derivatives, e.g. for elastic constants, are not implemented
    E(ε1, ε2) = d3_energy((1 + ε1 + ε2) * lattice, positions, DFTD3(RationalDamping("pbe")))
    @test_throws ErrorException ForwardDiff.derivative(0.0) do ε1
        ForwardDiff.derivative(ε2 -> E(ε1, ε2), 0.0)
    end
end

@testitem "DFTD3 in an SCF" tags=[:forces] begin
    using DFTK
    using SimpleDFTD3
    using LinearAlgebra

    # The D3 contribution to energy, forces and stresses, compared to SimpleDFTD3 directly
    lattice   = Matrix(6.0I, 3, 3)
    atoms     = fill(ElementCoulomb(:H), 2)
    positions = [[0.0, 0.0, 0.0], [0.12, 0.05, 0.03]]
    function run_scf(extra_terms)
        model = model_DFT(lattice, atoms, positions; functionals=LDA(), extra_terms,
                          symmetries=false)
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
