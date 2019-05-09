include("testcases_silicon.jl")

@testset "FFT and IFFT are an identity" begin
    Ecut = 4.0  # Hartree
    pw = PlaneWaveBasis(lattice, kpoints, Ecut)

    # Test X_to_Yst and then Yst_to_X, then X_to_Yst
    begin
        f_X = Array{ComplexF64}(randn(Float64, size(pw.Gs)))

        f_Yst = Array{ComplexF64}(undef, size(pw.grid_Yst))
        X_to_Yst!(pw, f_X, f_Yst)

        f2_X = similar(f_X)
        Yst_to_X!(pw, copy(f_Yst), f2_X)

        f2_Yst = similar(f_Yst)
        X_to_Yst!(pw, f2_X, f2_Yst)

        @test maximum(abs.(f2_X - f_X)) < 1e-12
        @test maximum(abs.(f2_Yst - f_Yst)) < 1e-12
    end
end
