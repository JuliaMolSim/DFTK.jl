@testitem "FFT and IFFT are an identity" setup=[TestCases] begin
    using DFTK
    using DFTK: PlaneWaveBasis, ifft!, fft!, ifft, fft
    using LinearAlgebra
    silicon = TestCases.silicon

    model = Model(silicon.lattice)
    pw    = PlaneWaveBasis(model; Ecut=4.0, fft_size=(8, 8, 8))

    @testset "Transformation on the cubic basis set" begin
        f_G = Array{ComplexF64}(randn(Float64, pw.fft_size...))

        f_R = Array{ComplexF64}(undef, pw.fft_size...)
        ifft!(f_R, pw, f_G)

        f2_G = fft(pw, f_R)
        f2_R = ifft(pw, f2_G)
        f3_G = fft!(similar(f_R), pw, f_R)

        @test maximum(abs.(f2_G - f_G)) < 1e-12
        @test maximum(abs.(f2_R - f_R)) < 1e-12
        @test maximum(abs.(f3_G - f_G)) < 1e-12

        ifft_mat = DFTK.ifft_matrix(pw)
        fft_mat = DFTK.fft_matrix(pw)
        @test maximum(abs.(ifft_mat * fft_mat - I)) < 1e-12
        @test maximum(abs.(ifft_mat * vec(f_G) - vec(f_R))) < 1e-12
        @test maximum(abs.(fft_mat * vec(f_R) - vec(f_G))) < 1e-12
    end

    @testset "Transformation on the spherical basis set" begin
        kpt = pw.kpoints[2]
        f_G = Array{ComplexF64}(randn(Float64, length(G_vectors(pw, kpt))))

        f_R = Array{ComplexF64}(undef, pw.fft_size...)
        ifft!(f_R, pw, kpt, f_G)

        f2_G = similar(f_G)
        fft!(f2_G, pw, kpt, copy(f_R))  # copy needed, because fft! destructive

        f2_R = similar(f_R)
        ifft!(f2_R, pw, kpt, f2_G)

        @test maximum(abs.(f2_G - f_G)) < 1e-12
        @test maximum(abs.(f2_R - f_R)) < 1e-12
    end
end
