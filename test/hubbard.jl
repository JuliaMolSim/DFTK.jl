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
   Ni = ElementPsp(:Ni, pseudopotentials)
   O = ElementPsp(:O, pseudopotentials)
   atoms = [Ni, O, Ni, O]
   positions = [[0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
                [0.75, 0.75, 0.75]]     
   magnetic_moments = [2, 0, -1, 0]

   # Hubbard parameters
   U        = 10u"eV"
   manifold = OrbitalManifold(atoms, Ni, "3D")
   
   model = model_DFT(lattice, atoms, positions; 
                     extra_terms=[Hubbard(manifold, U)],
                     temperature=0.01, functionals=PBE(),
                     smearing=DFTK.Smearing.Gaussian(), magnetic_moments=magnetic_moments)
   basis = PlaneWaveBasis(model; Ecut = 15, kgrid = [2, 2, 2])
   ρ0 = guess_density(basis, magnetic_moments)
   scfres = self_consistent_field(basis; tol=1e-10, ρ=ρ0)
   
   @testset "Test Energy results" begin
        # The reference values are obtained with first released version
        # of the Hubbard code and are in good agreement with Quantum Espresso
        ref = -354.907446880021
        e_total = scfres.energies.total
        @test e_total ≈ ref
        ref_hub = 0.17629078433258719
        @test scfres.energies.Hubbard ≈ ref_hub
   end
   # The unfolding of the kpoints is not supported with MPI
   if mpi_nprocs() == 1
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
