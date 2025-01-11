# This file collects examples, where we had issues with symmetries (symmetry determination,
# k-point reduction, etc.) which are now resolved. Should make sure we don't reintroduce bugs.

@testitem "Symmetry issues" setup=[TestCases] begin
    using DFTK
    using DFTK: spglib_dataset
    using Unitful
    using UnitfulAtomic
    using AtomsBase

    @testset "CuO2" begin
        a = 4.474
        lattice = [[0, a, a], [a, 0, a], [a, a, 0]]u"bohr"
        x = 6.711
        y = 2.237
        atoms = [
            Atom(:Cu, [0, 0, 0]u"bohr", magnetic_moment=0),
            Atom(:O,  [x, y, x]u"bohr", magnetic_moment=0),
            Atom(:O,  [x, y, y]u"bohr", magnetic_moment=0),
        ]
        system = periodic_system(atoms, lattice)

        symmetries = DFTK.symmetry_operations(system; check_symmetry=true)
        @test length(symmetries) == 48
        @test spglib_dataset(system).spacegroup_number == 225
    end

    @testset "Inlining" begin
        using PseudoPotentialData

        # Test that the index_G_vectors function is properly inlined by comparing timing
        # with a locally defined function known not to be inlined. Issue initially tackled
        # in PR https://github.com/JuliaMolSim/DFTK.jl/pull/1025
        function index_G_vectors_slow(basis, G::AbstractVector{<:Integer})
            start = .- cld.(basis.fft_size .- 1, 2)
            stop  = fld.(basis.fft_size .- 1, 2)
            lengths = stop .- start .+ 1

            # FFTs store wavevectors as [0 1 2 3 -2 -1] (example for N=5)
            function G_to_index(length, G)
                G >= 0 && return 1 + G
                return 1 + length + G
            end
            if all(start .<= G .<= stop)
                CartesianIndex(Tuple(G_to_index.(lengths, G)))
            else
                nothing  # Outside range of valid indices
            end
        end

        # This is a bare-bone version of the accumulate_over_symmetries() function, only
        # keeping calls to the index_G_vectors() function for which we test inlining
        function G_vectors_calls(basis, test_func)
            for symop in basis.symmetries
                invS = Mat3{Int}(inv(symop.S))
                for (ig, G) in enumerate(DFTK.G_vectors_generator(basis.fft_size))
                    igired = test_func(basis, invS * G)
                end
            end
        end

        silicon = TestCases.silicon
        Si = ElementPsp(silicon.atnum, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
        atoms = [Si, Si]
        model = model_DFT(silicon.lattice, atoms, silicon.positions;
                          functionals=[:lda_x, :lda_c_vwn])
        Ecut = 32
        kgrid = [1, 1, 1]
        basis = PlaneWaveBasis(model; Ecut, kgrid)

        actual_alloc = @allocated G_vectors_calls(basis, DFTK.index_G_vectors)
        slow_alloc = @allocated G_vectors_calls(basis, index_G_vectors_slow)
        @test slow_alloc > actual_alloc
    end
end

