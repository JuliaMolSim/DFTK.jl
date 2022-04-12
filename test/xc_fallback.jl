using Test
using DFTK: FallbackFunctional, LibxcFunctional, kernel_terms, potential_terms

@testset "Fallback LDA" begin
    for func_name in (:lda_x, :lda_c_vwn, :lda_c_pw)
        @testset "$func_name" begin
            fallback = FallbackFunctional{:lda}(func_name)
            libxc    = LibxcFunctional{:lda}(func_name)

            # Create reference
            ρ   = abs.(randn(1, 1, 1, 100))
            ref = kernel_terms(libxc, ρ)

            # Compute in DFTK in elevated precision
            ρbig  = Array{BigFloat}(ρ)
            terms = kernel_terms(fallback, ρbig)

            @test terms.zk     ≈ ref.zk     atol=5e-15
            @test terms.vrho   ≈ ref.vrho   atol=5e-15
            @test terms.v2rho2 ≈ ref.v2rho2 atol=5e-15

            # TODO Check floating-point type consistency:
            # ε = energy_per_particle(Val(func_name), rand(Float32))
            # @test Float32 == eltype(ε)
        end
    end
end

@testset "Fallback GGA" begin
    gga_fallback = (
        :gga_x_pbe,     :gga_c_pbe,     :gga_x_pbe_r,
        :gga_x_pbe_sol, :gga_c_pbe_sol,
        :gga_x_pbefe,   :gga_c_pbefe,
        :gga_x_xpbe,    :gga_c_xpbe,
        :gga_x_pbe_mol, :gga_c_pbe_mol,
        :gga_x_apbe,    :gga_c_apbe,
    )

    for func_name in gga_fallback
        @testset "$func_name" begin
            fallback = FallbackFunctional{:gga}(func_name)
            libxc    = LibxcFunctional{:gga}(func_name)

            # Create reference
            ρ   = abs.(randn(1, 1, 1, 100))
            σ   = abs.(randn(1, 1, 1, 100))
            ref = potential_terms(libxc, ρ, σ)

            # Compute in DFTK in elevated precision
            ρbig  = Array{BigFloat}(ρ)
            σbig  = Array{BigFloat}(σ)
            terms = potential_terms(fallback, ρ, σ)

            @test terms.zk     ≈ ref.zk     atol=5e-15
            @test terms.vrho   ≈ ref.vrho   atol=5e-15
            @test terms.vsigma ≈ ref.vsigma atol=5e-15

            # TODO Check floating-point type consistency:
            # ε = energy_per_particle(Val(func_name), rand(Float32))
            # @test Float32 == eltype(ε)
        end
    end
end
