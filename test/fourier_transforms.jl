using Test
using DFTK: PlaneWaveModel, G_to_r!, r_to_G!, G_to_r, r_to_G

include("testcases.jl")

@testset "FFT and IFFT are an identity" begin
    Ecut = 4.0  # Hartree
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, silicon.n_electrons)
    pw = PlaneWaveModel(model, fft_size, Ecut, silicon.kcoords,
                        silicon.kweights, silicon.ksymops)

    @testset "Transformation C_ρ <-> C_ρ^∗, 1 vector" begin
        f_G = Array{ComplexF64}(randn(Float64, pw.fft_size...))

        f_R = Array{ComplexF64}(undef, pw.fft_size...)
        G_to_r!(f_R, pw, f_G)

        f2_G = r_to_G(pw, f_R)
        f2_R = G_to_r(pw, f2_G)
        f3_G = r_to_G!(similar(f_R), pw, f_R)

        @test maximum(abs.(f2_G - f_G)) < 1e-12
        @test maximum(abs.(f2_R - f_R)) < 1e-12
        @test maximum(abs.(f3_G - f_G)) < 1e-12
    end

    @testset "Transformation C_ρ <-> C_ρ^∗, 3 vectors" begin
        f_G = Array{ComplexF64}(randn(Float64, pw.fft_size..., 3))

        f_R = Array{ComplexF64}(undef, pw.fft_size..., 3)
        G_to_r!(f_R, pw, f_G)

        f2_G = r_to_G(pw, f_R)
        f2_R = G_to_r(pw, f2_G)
        f3_G = r_to_G!(similar(f_R), pw, f_R)

        @test maximum(abs.(f2_G - f_G)) < 1e-12
        @test maximum(abs.(f2_R - f_R)) < 1e-12
        @test maximum(abs.(f3_G - f_G)) < 1e-12
    end

    @testset "Transformation B_k <-> C_ρ^∗, 1 vector" begin
        kpt = pw.kpoints[2]
        f_G = Array{ComplexF64}(randn(Float64, length(kpt.basis)))

        f_R = Array{ComplexF64}(undef, pw.fft_size...)
        G_to_r!(f_R, pw, kpt, f_G)

        f2_G = similar(f_G)
        r_to_G!(f2_G, pw, kpt, copy(f_R))  # copy needed, because r_to_G! destructive

        f2_R = similar(f_R)
        G_to_r!(f2_R, pw, kpt, f2_G)

        @test maximum(abs.(f2_G - f_G)) < 1e-12
        @test maximum(abs.(f2_R - f_R)) < 1e-12
    end

    @testset "Transformation B_k <-> C_ρ^∗, 3 vectors" begin
        kpt = pw.kpoints[2]
        f_G = Array{ComplexF64}(randn(Float64, length(kpt.basis), 3))

        f_R = Array{ComplexF64}(undef, pw.fft_size..., 3)
        G_to_r!(f_R, pw, kpt, f_G)

        f2_G = similar(f_G)
        r_to_G!(f2_G, pw, kpt, copy(f_R))  # copy needed, because r_to_G! destructive

        f2_R = similar(f_R)
        G_to_r!(f2_R, pw, kpt, f2_G)

        @test maximum(abs.(f2_G - f_G)) < 1e-12
        @test maximum(abs.(f2_R - f_R)) < 1e-12
    end
end
