using Test
using DFTK
using Plots
using QuadGK: quadgk
using SpecialFunctions: besselj
using BenchmarkTools: @benchmark
using Unitful 
using UnitfulAtomic
using LinearAlgebra
using StaticArrays

@testset "Check reading" begin
    psp = parse_tm_file("/Users/jasonlehto/git/Dino.jl/dev/DFTK/data/psp/tm/lda/tm-q15.pspnc")

    @test occursin(r"Tm", psp.description)
    @test occursin(r"Troullier-Martins", psp.description)
    @test psp.Zion == 69
    @test psp.valenceElectrons == 15
    @test psp.lmax == 3
    @test psp.lloc == 0
    @test psp.numGridPoints == 2001
    @test psp.r2well == 0.0
    @test psp.numProjectorFctns == [0,1,1,1]
    @test psp.pspCoreRadius ≈ [2.7274861, 3.4157018, 2.5944632, 2.7965336]
    @test psp.rms ≈ zeros(4)
    @test psp.energiesKB ≈ [zeros(2) for _ in 1:4]
    @test psp.epsatm ≈ zeros(4)
    @test psp.rchrg ≈ 3.70189202588749
    @test psp.fchrg ≈ 0.04664758709060
    @test psp.totCoreChrg ≈ 2.06513006799856
    @test sum(length(vals) for vals in psp.pseudoPotentials) ≈ 4 * 2001
    @test sum(length(vals) for vals in psp.firstProjectorVals) ≈ 4 * 2001
    @test isempty(psp.secondProjectorVals)
end

# plt = plot(psp.radialGrid, x -> eval_psp_semilocal_real(psp,x,0))
# display(plt)
@testset "Functions work" begin
    T = T where {T <: Real}
    psp = parse_tm_file("/Users/jasonlehto/git/Dino.jl/dev/DFTK/data/psp/tm/lda/h-q1.pspnc")
    r = psp.radialGrid[30]
    l = psp.lmax
   @test normalizer(psp,0)                          isa T
   @test eval_psp_local_real(psp,r)                 isa T
   @test eval_psp_local_fourier(psp,r)              isa T
   @test eval_pseudoWaveFunction_real(psp,r,l)      isa T
   @test eval_psp_semilocal_real(psp, r, l)         isa T
   @test normalizer(psp,l)                          isa T
   @test eval_psp_projector_fourier(psp, 1, l, 0.1) isa T
   @test eval_psp_projector_real(psp, 1, l, r)      isa T
end

@testset "Numerical integration to obtain fourier-space projectors" begin
    function integrand(psp, i, l, q, r) #See Lines: 802,811-813 in [m_psp1](https://github.com/abinit/abinit/blob/master/src/64_psp/m_psp1.F90)
        qr = 2π * q * r
        # @show eval_psp_projector_real(psp, i, l, r)
        bess = if l == 0
            -besselj(l, qr)
        else
            besselj(l - 1, qr) - (l + 1) * besselj(l, qr)/qr
        end
        return 2π * r^2 * eval_psp_projector_real(psp, i, l, r) * bess
    end

    dr,rmax = 0.01,10
    # for file in list_psp(family = "tm",datadir_psp = "/Users/jasonlehto/git/Dino.jl/dev/DFTK/data/psp/")
    #     psp = parse_tm_file("/Users/jasonlehto/git/Dino.jl/dev/DFTK/data/psp/" * file.identifier)
    #     @show psp.atomicNumber
    #     for (l,i) in zip([i for i in 0:psp.lmax],psp.numProjectorFctns)
    #         @show l,i
    #         l > length(psp.pspCoreRadius) - 1 && continue
    #         for q in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 100]                
    #             reference = quadgk(r -> integrand(psp, i, l, q, r), psp.radialGrid[1], psp.radialGrid[end]; rtol = 1e-5, atol = 1e-8) |> first
    #             val = eval_psp_projector_fourier(psp, i, l, q)
    #             @show q
    #             @test reference ≈ val atol=1e-8 rtol = 1e-5
    #             # @test reference ≈ eval_psp_projector_fourier(psp, q)
    #         end
    #     end
    # end
end

psp = parse_tm_file("dev/DFTK/data/psp/tm/lda/si-q4.pspnc")

# display(@benchmark eval_psp_local_fourier(psp, norm([-12,15,3])))

# @testset "Comparing Pseudopotential against known TM pseudopotentials" begin
Si = ElementPsp(14,:Si,parse_tm_file("dev/DFTK/data/psp/tm/lda/si-q4.pspnc"))
lattice = austrip(0.5431u"nm") / 2 * [  [0 1 1.];
                                    [1 0 1.];
                                    [1 1 0.]]
atoms = [Si => [ones(3)/8, -ones(3)/8]]
model = model_LDA(lattice,atoms)
kgrid = [4,4,4]
Ecut = 20
basis = PlaneWaveBasis(model, Ecut; kgrid = kgrid)
scfres = self_consistent_field(basis)
@show scfres.energies.total