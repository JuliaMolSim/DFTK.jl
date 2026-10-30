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
   @testset "Missing hubbard_n is an error" begin
        @test_throws ArgumentError DFTK.ene_ops(hubbard(basis), basis, scfres.ψ, scfres.occupation)
        @test_throws ArgumentError energy_hamiltonian(basis, nothing, nothing; scfres.ρ)
   end
   # The unfolding of the kpoints is not supported with MPI
   if mpi_nprocs(basis.comm_kpts) == 1
        @testset "Test symmetry consistency" begin
             n_hub = scfres.hubbard_n
             scfres_nosym = DFTK.unfold_bz(scfres)
             nhub_nosym = DFTK.compute_hubbard_n(scfres_nosym.basis, scfres_nosym.ψ,
                                                 scfres_nosym.occupation)
             @test n_hub ≈ nhub_nosym
        end
   end
end

@testitem "Atomic orbital projector derivatives" setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    silicon = TestCases.silicon

    # atomic_orbital_projectors: returns atomic projectors Φk in plane wave basis.
    # atomic_orbital_projector_derivatives(basis): returns derivatives of atomic projectors
    #                                              dΦk/dR in plane wave basis.
    # atomic_orbital_projector_derivatives(basis, ψ): returns the overlap ψ' * dΦk/dR.

    Si = ElementPsp(:Si, TestCases.pd_lda_family)
    positions = [silicon.positions[1] + [0.02, -0.01, 0.0], silicon.positions[2]]
    model = model_atomic(silicon.lattice, [Si, Si], positions; symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=10, kgrid=(2, 2, 2))

    dir = normalize(randn(3))
    h = 1e-5
    function projectors_at(ε)
        positions_ε = [positions[1] + ε * dir, positions[2]]
        DFTK.atomic_orbital_projectors(basis; positions=positions_ε).projectors
    end
    reference = (projectors_at(h) - projectors_at(-h)) / 2h

    ψ = [randn(ComplexF64, size(P, 1), 4) for P in projectors_at(0.0)]
    dprojectors = DFTK.atomic_orbital_projector_derivatives(basis).dprojectors
    ψ_dprojectors = DFTK.atomic_orbital_projector_derivatives(basis, ψ).dprojectors
    for ik in eachindex(basis.kpoints)
        # All columns change, also those of the projectors on atom 2
        @test sum(dir .* dprojectors[ik][1]) ≈ reference[ik] rtol=1e-6
        @test sum(dir .* ψ_dprojectors[ik][1]) ≈ ψ[ik]' * reference[ik] rtol=1e-6
    end
end
