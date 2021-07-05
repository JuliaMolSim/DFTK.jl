using Test
using DFTK: PlaneWaveBasis, G_to_r!, r_to_G!, G_to_r, r_to_G

include("testcases.jl")

@testset "FFT and IFFT are an identity" begin
    Ecut = 4.0  # Hartree
    fft_size = [8, 8, 8]
    model = Model(silicon.lattice, n_electrons=silicon.n_electrons)
    pw = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    @testset "Transformation on the cubic basis set" begin
        f_G = Array{ComplexF64}(randn(Float64, pw.fft_size...))

        f_R = Array{ComplexF64}(undef, pw.fft_size...)
        G_to_r!(f_R, pw, f_G)

        f2_G = r_to_G(pw, f_R)
        f2_R = G_to_r(pw, f2_G; assume_real=false)
        f3_G = r_to_G!(similar(f_R), pw, f_R)

        @test maximum(abs.(f2_G - f_G)) < 1e-12
        @test maximum(abs.(f2_R - f_R)) < 1e-12
        @test maximum(abs.(f3_G - f_G)) < 1e-12

        G_to_r_mat = DFTK.G_to_r_matrix(pw)
        r_to_G_mat = DFTK.r_to_G_matrix(pw)
        @test maximum(abs.(G_to_r_mat * r_to_G_mat - I)) < 1e-12
        @test maximum(abs.(G_to_r_mat * vec(f_G) - vec(f_R))) < 1e-12
        @test maximum(abs.(r_to_G_mat * vec(f_R) - vec(f_G))) < 1e-12
    end

    @testset "Transformation on the spherical basis set" begin
        kpt = pw.kpoints[2]
        f_G = Array{ComplexF64}(randn(Float64, length(G_vectors(kpt))))

        f_R = Array{ComplexF64}(undef, pw.fft_size...)
        G_to_r!(f_R, pw, kpt, f_G)

        f2_G = similar(f_G)
        r_to_G!(f2_G, pw, kpt, copy(f_R))  # copy needed, because r_to_G! destructive

        f2_R = similar(f_R)
        G_to_r!(f2_R, pw, kpt, f2_G)

        @test maximum(abs.(f2_G - f_G)) < 1e-12
        @test maximum(abs.(f2_R - f_R)) < 1e-12
    end
end
