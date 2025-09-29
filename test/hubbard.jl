@testitem "Test Hubbard U term in Nickel Oxide" setup=[TestCases] begin
   using DFTK
   using PseudoPotentialData
   using Unitful
   using UnitfulAtomic
   using LinearAlgebra
   
   # Hubbard parameters
   U        = 10u"eV"
   manifold = DFTK.OrbitalManifold(;species=:Ni, label="3D")
   
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
   
   model = model_DFT(lattice, atoms, positions; 
                     extra_terms=[DFTK.Hubbard(manifold, U)],
                     temperature=0.01, functionals=PBE(),
                     smearing=DFTK.Smearing.Gaussian(), magnetic_moments=magnetic_moments)
   basis = PlaneWaveBasis(model; Ecut = 15, kgrid = [2, 2, 2])
   ρ0 = guess_density(basis, magnetic_moments)
   scfres = self_consistent_field(basis; tol=1e-10, ρ=ρ0)
   
   ref = -354.907446880021
   e_total = scfres.energies.total
   @test abs(e_total - ref) < 1e-8
   for (term, value) in scfres.energies
       if term == "Hubbard"
          ref_hub = 0.17629078433258719
          e_hub = value
          @test abs(e_hub - ref_hub) < 1e-8
       end
   end  

   # Test symmetry consistency
   n_hub = scfres.nhubbard
   basis_nosym = DFTK.unfold_bz(basis)
   ρ0 = guess_density(basis_nosym, magnetic_moments)
   scfres_nosym = self_consistent_field(basis_nosym; tol=1e-10, ρ=ρ0)
   @test norm(n_hub .- scfres_nosym.nhubbard) < 1e-8

end

@testitem "Test Wigner matrices on Silicon symmetries" setup=[TestCases] begin
   using DFTK
   using PseudoPotentialData
   using LinearAlgebra

   lattice =  [[0 1 1.];
               [1 0 1.];
               [1 1 0.]]
   positions = [ones(3)/8, -ones(3)/8]
   pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
   Si = ElementPsp(:Si, pseudopotentials)
   atoms = [Si, Si]
   model = model_DFT(lattice, atoms, positions; functionals=PBE())
   basis = PlaneWaveBasis(model; Ecut=32, kgrid=[2, 2, 2])

   D = DFTK.Wigner_sym(1, lattice, basis.symmetries)
   D5 = [-1 0 0; 0 -1 0; 0 0 1]
   @test norm(D[:,:,1] - I) < 1e-8
   @test norm(D[:,:,25] + I) < 1e-8
   @test norm(D[:,:,5] - D5) < 1e-8
end 