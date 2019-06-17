include("testcases_silicon.jl")

@testset "FFT and IFFT are an identity" begin
    Ecut = 4.0  # Hartree
    grid_size = [15, 15, 15]
    pw = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights)

    @testset "Density grid transformation" begin
        f_G = Array{ComplexF64}(randn(Float64, prod(pw.grid_size)))

        f_R = Array{ComplexF64}(undef, size(pw.FFT)...)
        DFTK.G_to_r!(pw, f_G, f_R)

        f2_G = similar(f_G)
        DFTK.r_to_G!(pw, copy(f_R), f2_G)

        f2_R = similar(f_R)
        DFTK.G_to_r!(pw, f2_G, f2_R)

        @test maximum(abs.(f2_G - f_G)) < 1e-12
        @test maximum(abs.(f2_R - f_R)) < 1e-12
    end

    @testset "Wave function grid transformation" begin
        ik = 2
        f_G = Array{ComplexF64}(randn(Float64, size(pw.basis_wf[ik])))

        f_R = Array{ComplexF64}(undef, size(pw.FFT)...)
        DFTK.G_to_r!(pw, f_G, f_R, gcoords=pw.basis_wf[ik])

        f2_G = similar(f_G)
        DFTK.r_to_G!(pw, copy(f_R), f2_G, gcoords=pw.basis_wf[ik])

        f2_R = similar(f_R)
        DFTK.G_to_r!(pw, f2_G, f2_R, gcoords=pw.basis_wf[ik])

        @test maximum(abs.(f2_G - f_G)) < 1e-12
        @test maximum(abs.(f2_R - f_R)) < 1e-12
    end
end
