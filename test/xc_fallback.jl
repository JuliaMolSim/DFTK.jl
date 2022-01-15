using Test
using Libxc
using DFTK: xc_fallback!

@testset "Fallback LDA" begin
    for func_name in (:lda_x, :lda_c_vwn)
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
