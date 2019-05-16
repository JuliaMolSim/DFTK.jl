include("testcases_silicon.jl")

function test_pw_cutoffs(lattice, kpoints, kweights, Ecut, supersampling)
    pw = PlaneWaveBasis(lattice, kpoints, kweights, Ecut,
                        supersampling_Y=supersampling)

    for (ik, k) in enumerate(kpoints)
        for ig in pw.kmask[ik]
            @test sum(abs2, k + pw.Gs[ig]) ≤ 2 * Ecut
        end
    end
end

@testset "Check with reference data" begin
    # TODO Extend this test

    Ecut = 3
    pw = PlaneWaveBasis(lattice, kpoints, kweights, Ecut)

    @test pw.lattice == lattice
    @test pw.recip_lattice ≈ 2π * inv(lattice)
    @test pw.unit_cell_volume ≈ det(lattice)

    @test pw.kpoints == kpoints
    @test pw.kweights == kweights / sum(kweights)
    # @test pw.Gs
    @test pw.Ecut == 3
    @test pw.Gs[pw.idx_DC] == zeros(3)
    # @test pw.kmask
    # @test pw.qsq

    # @test pw.grid_Yst
    # @test pw.idx_to_fft
end

@testset "Energy cutoff is respected" begin
    test_pw_cutoffs(lattice, kpoints, kweights, 4.0, 2)
    test_pw_cutoffs(lattice, kpoints, kweights, 3.0, 2)
    test_pw_cutoffs(lattice, kpoints, kweights, 4.0, 1.8)
end
