using Test
using Libxc
using DFTK: xc_fallback!

@testset "Fallback LDA" begin
    for func_name in (:lda_x, :lda_c_vwn, :lda_c_pw)
        func = Libxc.Functional(func_name)

        # Create reference
        ρ = abs.(randn(100))
        Eref  = similar(ρ)
        Vref  = similar(ρ)
        V2ref = similar(ρ)
        Libxc.evaluate!(func, rho=ρ, zk=Eref, vrho=Vref, v2rho2=V2ref)

        # Compute in DFTK in elevated precision
        ρbig = Array{BigFloat}(ρ)
        E    = similar(ρbig)
        V    = similar(ρbig)
        V2   = similar(ρbig)
        xc_fallback!(func, Val(:lda), ρbig; zk=E, vrho=V, v2rho2=V2)

        @test E  ≈ Eref  atol=5e-15
        @test V  ≈ Vref  atol=5e-15
        @test V2 ≈ V2ref atol=5e-15
    end
end

@testset "Fallback GGA" begin
    for func_name in (:gga_x_pbe, :gga_c_pbe)
        func = Libxc.Functional(func_name)

        # Create reference
        ρ = abs.(randn(100))
        σ = abs.(randn(100))
        Eref  = similar(ρ)
        Vρref = similar(ρ)
        Vσref = similar(ρ)
        Libxc.evaluate!(func, rho=ρ, sigma=σ, zk=Eref, vrho=Vρref, vsigma=Vσref)

        # Compute in DFTK in elevated precision
        ρbig = Array{BigFloat}(ρ)
        σbig = Array{BigFloat}(σ)
        E    = similar(ρbig)
        Vρ   = similar(ρbig)
        Vσ   = similar(ρbig)
        xc_fallback!(func, Val(:gga), ρbig; sigma=σbig, zk=E, vrho=Vρ, vsigma=Vσ)

        @test E  ≈ Eref   atol=5e-15
        @test Vρ ≈ Vρref  atol=5e-15
        @test Vσ ≈ Vσref  atol=5e-15
    end
end
