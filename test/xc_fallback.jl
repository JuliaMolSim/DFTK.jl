using Test
using DFTK: lda_x!, lda_c_vwn!
using Libxc

@testset "Fallback lda_x!" begin
    func = Libxc.Functional(:lda_x)

    # Create reference
    ρ = abs.(randn(100))
    Eref = similar(ρ)
    Vref = similar(ρ)
    Libxc.evaluate!(func, rho=ρ, zk=Eref, vrho=Vref)

    # Compute in DFTK in elevated precision
    ρbig = Array{BigFloat}(ρ)
    E=similar(ρbig)
    V=similar(ρbig)
    lda_x!(ρbig, E=E, Vρ=V)

    @test E ≈ Eref atol=5e-15
    @test V ≈ Vref atol=5e-15
end

@testset "Fallback lda_c_vwn!" begin
    func = Libxc.Functional(:lda_c_vwn)

    # Create reference
    ρ = abs.(randn(100))
    Eref = similar(ρ)
    Vref = similar(ρ)
    Libxc.evaluate!(func, rho=ρ, zk=Eref, vrho=Vref)

    # Compute in DFTK in elevated precision
    ρbig = Array{BigFloat}(ρ)
    E=similar(ρbig)
    V=similar(ρbig)
    lda_c_vwn!(ρbig, E=E, Vρ=V)

    @test E ≈ Eref atol=5e-15
    @test V ≈ Vref atol=5e-15
end
