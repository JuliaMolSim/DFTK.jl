include("testcases_silicon.jl")

@testset "FFT and IFFT are an identity" begin
    Ecut = 4.0  # Hartree
    pw = PlaneWaveBasis(lattice, kpoints, kweights, Ecut)

    # Test Y_to_Yst and then Yst_to_Y, then Y_to_Yst
    begin
        f_Y = Array{ComplexF64}(randn(Float64, size(pw.Gs)))

        f_Yst = Array{ComplexF64}(undef, size(pw.grid_Yst))
        Y_to_Yst!(pw, f_Y, f_Yst)

        f2_Y = similar(f_Y)
        Yst_to_Y!(pw, copy(f_Yst), f2_Y)

        f2_Yst = similar(f_Yst)
        Y_to_Yst!(pw, f2_Y, f2_Yst)

        @test maximum(abs.(f2_Y - f_Y)) < 1e-12
        @test maximum(abs.(f2_Yst - f_Yst)) < 1e-12
    end
end
