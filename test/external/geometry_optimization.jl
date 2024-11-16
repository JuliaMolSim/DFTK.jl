@testitem "Test Geometry optimization" tags=[:atomsbase] begin
    using DFTK
    using AtomsBase
    using PseudoPotentialData
    using GeometryOptimization
    using Unitful
    using UnitfulAtomic
    GO = GeometryOptimization

    calc = DFTKCalculator(;
        model_kwargs=(;
            functionals=LDA(),
            pseudopotentials=PseudoFamily("pd_nc_sr_lda_standard_0.4.1_upf")
        ),
        basis_kwargs=(; kgrid=[1, 1, 1], Ecut=10)  # Crude numerical parameters
    )

    # Initial hydrogen molecule
    r0 = 1.4   # Initial bond length in Bohr
    a  = 10.0  # Box size in Bohr

    bounding_box = [[a, 0.0, 0.0],
                    [0.0, a, 0.0],
                    [0.0, 0.0, a]]u"bohr"
    h2_crude = periodic_system([:H => [0, 0, 0.]u"bohr", :H => [0, 0, r0]u"bohr"],
                               bounding_box)

    @testset "Testing against ABINIT reference" begin
        # determined at the same functional and Ecut
        # see https://docs.abinit.org/tutorial/base1/
        results = minimize_energy!(h2_crude, calc; tol_forces=2e-6)
        rmin = norm(position(results.system[1]) - position(results.system[2]))
        @test austrip(rmin) ≈ 1.486 atol=2e-3
    end
end
