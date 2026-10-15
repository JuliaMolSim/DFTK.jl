@testitem "Test Wigner matrices" setup=[TestCases] begin
   using DFTK
   using PseudoPotentialData
   using LinearAlgebra
   
    @testset "Wigner Identity" begin
        # Identity
        Id = [1.0 0 0; 0 1 0; 0 0 1]
        D = DFTK.wigner_d_matrix(1, Id)
        @test D ≈ I
        D = DFTK.wigner_d_matrix(2, Id)
        @test D ≈ I
    end
    @testset "Wigner Inversion" begin
        # This reverts all p orbitals, sends all d orbitals in themselves
        Inv = -[1.0 0 0; 0 1 0; 0 0 1]
        D = DFTK.wigner_d_matrix(1, Inv)
        @test D ≈ -I
        D = DFTK.wigner_d_matrix(2, Inv)
        @test D ≈ I
    end
    @testset "Wigner invert x and y" begin
        # This keeps pz, dz2, dx2-y2 and dxy unchanged, changes sign to all others
        A3  = [1.0 0 0; 0 -1 0; 0 0 -1] 
        D3p = [-1.0 0 0; 0 -1 0; 0 0 1]
        D3d = [-1.0 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 -1 0; 0 0 0 0 1]
        D = DFTK.wigner_d_matrix(1, A3)
        @test D ≈ D3p
        D = DFTK.wigner_d_matrix(2, A3)
        @test D ≈ D3d
    end
    @testset "Wigner swap x and y" begin
        # This sends: px <-> py, dxz <-> dyz, dx2-y2 -> -(dx2-y2) and keeps the other fixed
        A3  = [0.0 1 0; 1 0 0; 0 0 1] 
        D3p = [0.0 0 1; 0 1 0; 1 0 0]
        D3d = [1.0 0 0 0 0; 0 0 0 1 0; 0 0 1 0 0; 0 1 0 0 0; 0 0 0 0 -1]
        D = DFTK.wigner_d_matrix(1, A3)
        @test D ≈ D3p
        D = DFTK.wigner_d_matrix(2, A3)
        @test D ≈ D3d
    end
end

@testitem "OrbitalManifold construction" begin
    using AtomsBase
    using DFTK
    using PseudoPotentialData

    a = 7.9  # Bohr
    lattice = a * [[ 1.0  0.5  0.5];
                   [ 0.5  1.0  0.5];
                   [ 0.5  0.5  1.0]]
    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    Ni = ElementPsp(:Ni, pseudopotentials)
    O = ElementPsp(:O, pseudopotentials)
    atoms = [Ni, O, Ni, O]
    positions = [[0.0, 0.0, 0.0],
                 [0.25, 0.25, 0.25],
                 [0.5, 0.5, 0.5],
                 [0.75, 0.75, 0.75]]
    model = model_DFT(lattice, atoms, positions; functionals=PBE())

    # Test all supported ways of constructing an OrbitalManifold
    for manifold in [OrbitalManifold(Ni,                   "3D"),
                     OrbitalManifold(:Ni,                  "3D"),
                     OrbitalManifold(ChemicalSpecies(:Ni), "3D"),
                     OrbitalManifold(Ni,                   (; l=2, i=1)),
                     OrbitalManifold(:Ni,                  (; l=2, i=1)),
                     OrbitalManifold(ChemicalSpecies(:Ni), (; l=2, i=1))]
        resolved_manifold = DFTK.resolve_hubbard_manifold(manifold, model)

        @test resolved_manifold.psp === Ni.psp
        @test resolved_manifold.iatoms == [1, 3]
        @test resolved_manifold.l == 2
        @test resolved_manifold.i == 1
    end
end
 
@testitem "Test Hubbard U term in Nickel Oxide" setup=[TestCases] begin
   using DFTK
   using PseudoPotentialData
   using Unitful
   using UnitfulAtomic
   using LinearAlgebra
   
   a = 7.9  # Bohr
   lattice = a * [[ 1.0  0.5  0.5];
                  [ 0.5  1.0  0.5];
                  [ 0.5  0.5  1.0]]
   pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
   Ni = ElementPsp(:Ni, pseudopotentials; rcut=10.0)  # QuantumEspresso rcut value
   O = ElementPsp(:O, pseudopotentials; rcut=10.0)    # QuantumEspresso rcut value
   atoms = [Ni, O, Ni, O]
   positions = [[0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
                [0.75, 0.75, 0.75]]     
   magnetic_moments = [2, 0, -1, 0]

   # Hubbard parameters
   hubbard = Hubbard(OrbitalManifold(Ni, "3D") => 10u"eV")
   
   model = model_DFT(lattice, atoms, positions; 
                     extra_terms=[hubbard],
                     temperature=0.01, functionals=PBE(),
                     smearing=DFTK.Smearing.Gaussian(), magnetic_moments)
   basis = PlaneWaveBasis(model; Ecut = 15, kgrid = [2, 2, 2])
   ρ0 = guess_density(basis, magnetic_moments)
   scfres = self_consistent_field(basis; tol=1e-10, ρ=ρ0)
   
   @testset "Test Energy results" begin
        # The reference values are obtained with first released version
        # of the Hubbard code in DFTK and are in good reasonable agreement with Quantum Espresso
        # but at this supersampling the nlcc core density is badly aliased (~10mHa on the total energy),
        # and the comparison should not be trusted too much
        ref = -354.91625589331755
        e_total = scfres.energies.total
        @test e_total ≈ ref
        ref_hub = 0.17631957087331435
        @test scfres.energies.Hubbard ≈ ref_hub
   end
   # The unfolding of the kpoints is not supported with MPI
   if mpi_nprocs(basis.comm_kpts) == 1
        @testset "Test symmetry consistency" begin
             n_hub = scfres.hubbard_n
             scfres_nosym = DFTK.unfold_bz(scfres)
             term_idx = findfirst(term -> isa(term, DFTK.TermHubbard),
                                  scfres_nosym.basis.terms)
             term_hub = scfres_nosym.basis.terms[term_idx]
             nhub_nosym = DFTK.compute_hubbard_n(term_hub, scfres_nosym.basis,
                                                 scfres_nosym.ψ, scfres_nosym.occupation)
             @test n_hub ≈ nhub_nosym
        end
   end
end

@testitem "Löwdin orthogonalization ForwardDiff rule" tags=[:minimal] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra
    using Random

    δφ = randn(Xoshiro(2), ComplexF64, 8, 3)
    # Orthonormal columns (fully degenerate overlap) and generic non-orthogonal columns
    for φ in (Matrix(qr(randn(Xoshiro(1), ComplexF64, 8, 3)).Q), randn(Xoshiro(3), ComplexF64, 8, 3))
        f(t) = real(sum(DFTK.ortho_lowdin(φ + t * δφ) * [1.0, 2.0, 3.0 + 1im]))
        h = 1e-5
        @test ForwardDiff.derivative(f, 0.0) ≈ (f(h) - f(-h)) / 2h  rtol=1e-7
    end
end

@testmodule HubbardSilicon begin
    using DFTK
    using LinearAlgebra
    using Logging
    using Unitful
    using UnitfulAtomic
    using ..TestCases: silicon, pd_lda_family

    # Unphysical but cheap: U on Si 3P. The lattice is strained and the atoms are displaced
    # keeping the inversion symmetry, so that forces and stresses are non-zero, need to be
    # symmetrized, and the Löwdin orthogonalization couples both atoms.
    Si = ElementPsp(:Si, pd_lda_family)
    lattice = silicon.lattice * [1.0 0.0 0.0; 0.05 1.0 0.0; 0.0 0.0 1.02]
    p = silicon.positions[1] + [0.02, -0.01, 0.0]
    positions = [p, -p]
    hubbard_Si = Hubbard(OrbitalManifold(:Si, "3P") => 5u"eV")
    function make_basis(lattice, positions; symmetries=true, hubbard=hubbard_Si)
        model = model_DFT(lattice, [Si, Si], positions; functionals=LDA(), temperature=1e-2,
                          extra_terms=[hubbard], symmetries)
        with_logger(NullLogger()) do
            PlaneWaveBasis(model; Ecut=10, kgrid=(2, 2, 2))
        end
    end
    # Fixed directions, so that the Hubbard contributions are surely non-negligible
    dir_atom   = normalize([0.3, -0.7, 0.2])
    dir_strain = [0.3 0.2 -0.1; 0.2 -0.5 0.1; 0.0 0.4 0.7]
    function displaced(ε, iatom=1)
        [position + (i == iatom) * ε * dir_atom for (i, position) in enumerate(positions)]
    end
    strained(ε) = lattice + ε * dir_strain * lattice

    # Hubbard energy at fixed ψ; hubbard_n changes since the projectors move
    function hubbard_energy(ψ, occupation, lattice, positions; hubbard=hubbard_Si)
        basis = make_basis(lattice, positions; symmetries=false, hubbard)
        ihub = findfirst(term -> term isa DFTK.TermHubbard, basis.terms)
        hubbard_n = DFTK.compute_hubbard_n(basis, ψ, occupation)
        DFTK.ene_ops(basis.terms[ihub], basis, ψ, occupation; hubbard_n).E
    end
end

@testitem "Hubbard projector derivatives and term-wise forces and stresses" #=
    =#    tags=[:forces] setup=[TestCases, HubbardSilicon] begin
    # Derivatives of the Hubbard energy at fixed ψ, which hold for any ψ: no SCF is needed.
    using DFTK
    using ForwardDiff
    import FiniteDiff
    using LinearAlgebra
    using Random
    using Unitful
    using UnitfulAtomic
    using .HubbardSilicon: make_basis, lattice, positions, dir_atom, dir_strain,
                            displaced, strained, hubbard_energy

    basis = make_basis(lattice, positions; symmetries=false)
    # Orthonormal ψ with a large weight on the atomic orbitals, so that hubbard_n is sizeable
    projectors = DFTK.atomic_orbital_projectors(basis).projectors
    ψ = [DFTK.ortho_qr(Pk + 0.3randn(Xoshiro(ik), eltype(Pk), size(Pk)))
         for (ik, Pk) in enumerate(projectors)]
    occupation = [[2.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.0] for _ in basis.kpoints]

    @testset "Projector derivatives" begin
        h = 1e-5
        projectors_at(ε) = DFTK.atomic_orbital_projectors(basis; positions=displaced(ε)).projectors
        ref = (projectors_at(h) - projectors_at(-h)) / 2h
        dprojectors = DFTK.atomic_orbital_projector_derivatives(basis).dprojectors
        # Overlaps with the Bloch waves, computed without the plane-wave sized derivative
        ψresults = DFTK.atomic_orbital_projector_derivatives(basis; ψ)
        (; projections) = ψresults
        ψdprojectors = ψresults.dprojectors
        projectors = projectors_at(0.0)
        for ik in eachindex(basis.kpoints)
            @test sum(dir_atom .* dprojectors[ik][1]) ≈ ref[ik]  rtol=1e-6
            @test ψdprojectors[ik][1] ≈ [ψ[ik]' * dP for dP in dprojectors[ik][1]]
            @test projections[ik] ≈ ψ[ik]' * projectors[ik]
        end
    end

    # Unsorted atoms, and U on one atom only: the other atom feels a force through the
    # Löwdin orthogonalization
    @testset "Term-wise force, U on atoms $atoms" for atoms in ([1, 2], [2, 1], [1])
        hubbard = Hubbard(OrbitalManifold(atoms, "3P") => 5u"eV")
        basis_U = make_basis(lattice, positions; symmetries=false, hubbard)
        ihub = findfirst(term -> term isa DFTK.TermHubbard, basis_U.terms)
        forces = DFTK.compute_forces(basis_U.terms[ihub], basis_U, ψ, occupation)
        for iatom = 1:2
            dE = FiniteDiff.finite_difference_derivative(0.0) do ε
                hubbard_energy(ψ, occupation, lattice, displaced(ε, iatom); hubbard)
            end
            @test abs(dE) > 1e-4
            @test dot(dir_atom, forces[iatom]) ≈ -dE  atol=1e-8
        end
    end

    @testset "Term-wise strain derivative (AD through the Löwdin rule)" begin
        E(ε) = hubbard_energy(ψ, occupation, strained(ε), positions)
        dE = ForwardDiff.derivative(E, 0.0)
        @test abs(dE) > 1e-4
        @test dE ≈ FiniteDiff.finite_difference_derivative(E, 0.0)  atol=1e-8
    end
end

@testitem "Hubbard forces and stresses vs SCF energies" #=
    =#    tags=[:forces, :slow] setup=[TestCases, HubbardSilicon] begin
    using DFTK
    import FiniteDiff
    using LinearAlgebra
    using .HubbardSilicon: make_basis, lattice, positions, dir_atom, dir_strain,
                            displaced, strained

    function scf_energy(lattice, positions)
        basis = make_basis(lattice, positions; symmetries=false)
        self_consistent_field(basis; tol=1e-11).energies.total
    end
    scfres = self_consistent_field(make_basis(lattice, positions); tol=1e-11)
    scfres_nosym = DFTK.unfold_bz(scfres)
    @test length(scfres.basis.symmetries) > 1

    @testset "Total forces" begin
        forces = compute_forces(scfres)
        @test forces ≈ compute_forces(scfres_nosym)  atol=1e-8
        dE = FiniteDiff.finite_difference_derivative(ε -> scf_energy(lattice, displaced(ε)), 0.0)
        @test dot(dir_atom, forces[1]) ≈ -dE  atol=1e-7
    end

    @testset "Stresses" begin
        stresses = compute_stresses_cart(scfres)
        @test stresses ≈ compute_stresses_cart(scfres_nosym)  atol=1e-8
        dE = FiniteDiff.finite_difference_derivative(ε -> scf_energy(strained(ε), positions), 0.0)
        @test dot(dir_strain, stresses) * scfres.basis.model.unit_cell_volume ≈ dE  atol=1e-7
    end
end
