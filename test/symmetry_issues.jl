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

        # Test that the index_G_vectors function is properly inlined.
        # Issue initially tackled in PR https://github.com/JuliaMolSim/DFTK.jl/pull/1025

        # This is a bare-bone version of the accumulate_over_symmetries() function, only
        # keeping calls to the index_G_vectors() function for which we test inlining
        function G_vectors_calls(basis)
            for symop in basis.symmetries
                invS = Mat3{Int}(inv(symop.S))
                for (ig, G) in enumerate(DFTK.G_vectors_generator(basis.fft_size))
                    igired = DFTK.index_G_vectors(basis, invS * G)
                end
            end
        end

        # If a function is inlined, its name disappers from the output of code_typed(;optimize=true)
        # When optimize=false, the function is not inlined and we can see its name in the output.
        function is_inlined(optimize)
            info = code_typed(G_vectors_calls, (DFTK.PlaneWaveBasis,), optimize=optimize)
            codeinfo = first(info).first
            output = sprint(show, MIME"text/plain"(), codeinfo.code)
            return !occursin("index_G_vectors", output)
        end

        # Test that the index_G_vectors function is inlined
        @test is_inlined(true)
        @test !is_inlined(false)
    end
end

