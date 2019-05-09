include("testcases_silicon.jl")

function test_pw_cutoffs(lattice, kpoints, Ecut, supersampling)
    pw = PlaneWaveBasis(lattice, kpoints, Ecut,
                        supersampling_Y=supersampling)

    for (ik, k) in enumerate(kpoints)
        for ig in pw.kmask[ik]
            @test sum(abs2, k + pw.Gs[ig]) ≤ 2 * Ecut
        end
    end
end

@testset "Check with reference data" begin
    # TODO Check recip_lattice, unit_cell_volume, Gs, grid_Yst
    #      against some reference
end

@testset "Energy cutoff is respected" begin
    test_pw_cutoffs(lattice, kpoints, 4.0, 2)
    test_pw_cutoffs(lattice, kpoints, 3.0, 2)
    test_pw_cutoffs(lattice, kpoints, 4.0, 1.8)
end
