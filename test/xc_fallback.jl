using Test
using Libxc
using DFTK

@testset "Fallback LDA" begin
    for func_name in (:lda_x, :lda_c_vwn)
        func = Libxc.Functional(func_name)

        # Create reference
        ρ = abs.(randn(100))
        Eref = similar(ρ)
        Vref = similar(ρ)
        Libxc.evaluate!(func, rho=ρ, zk=Eref, vrho=Vref)

        # Compute in DFTK in elevated precision
        ρbig = Array{BigFloat}(ρ)
        E=similar(ρbig)
        V=similar(ρbig)
        Libxc.evaluate!(func, rho=ρbig, zk=E, vrho=V)

        @test E ≈ Eref atol=5e-15
        @test V ≈ Vref atol=5e-15
    end
end
