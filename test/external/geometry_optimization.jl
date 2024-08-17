@testitem "Test Geometry optimization" tags=[:atomsbase] begin
    using DFTK
    using GeometryOptimization
    using LinearAlgebra
    using Unitful
    using UnitfulAtomic
    using LazyArtifacts
    GO = GeometryOptimization

    calc = DFTKCalculator(;
        model_kwargs = (; functionals=LDA()),        # xc functionals
        basis_kwargs = (; kgrid=[1, 1, 1], Ecut=10)  # Crude numerical parameters
    )

    # Initial hydrogen molecule
    r0 = 1.4   # Initial bond length in Bohr
    a  = 10.0  # Box size in Bohr

    lattice = a * I(3)
    H = ElementPsp(:H; psp=load_psp(artifact"pd_nc_sr_pbe_standard_0.4.1_upf/H.upf"));
    atoms = [H, H]
    positions = [zeros(3), lattice \ [r0, 0., 0.]]
    h2_crude = periodic_system(lattice, atoms, positions)

    @testset "Testing against ABINIT reference" begin
        # determined at the same functional and Ecut
        # see https://docs.abinit.org/tutorial/base1/
        results = minimize_energy!(h2_crude, calc; tol_forces=2e-6)
        rmin = norm(position(results.system[1]) - position(results.system[2]))
        @test austrip(rmin) ≈ 1.486 atol=2e-3
    end
end
