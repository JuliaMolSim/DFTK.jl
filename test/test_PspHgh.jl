@testset "Check reading 'C-lda-q4'" begin
    psp = PspHgh("C-lda-q4")

    @test endswith(psp.path, "c-pade-q4.hgh")
    @test psp.identifier == "c-pade-q4"
    @test occursin("c", lowercase(psp.description))
    @test occursin("pade", lowercase(psp.description))
    @test psp.Zion == 4
    @test psp.rloc == 0.34883045
    @test psp.c == [-8.51377110, 1.22843203]
    @test psp.lmax == 1
    @test psp.rp == [0.30455321, 0.2326773]
    @test psp.h[1] == 9.52284179 * ones(1, 1)
    @test psp.h[2] == zeros(0, 0)
end

@testset "Check reading 'Ni-pade-q18'" begin
    psp = PspHgh("Ni-pade-q18")

    @test endswith(psp.path, "ni-pade-q18.hgh")
    @test psp.identifier == "ni-pade-q18"
    @test occursin("ni", lowercase(psp.description))
    @test occursin("pade", lowercase(psp.description))
    @test psp.Zion == 18
    @test psp.rloc == 0.35000000
    @test psp.c == [3.61031072, 0.44963832]
    @test psp.lmax == 2
    @test psp.rp == [0.24510489, 0.23474136, 0.21494950]
    @test psp.h[1] == [[12.16113071, 3.51625420] [3.51625420, -9.07892931]]
    @test psp.h[2] == [[-0.82062357, 2.54774737] [2.54774737, -6.02907069]]
    @test psp.h[3] == -13.39506212 * ones(1, 1)
end
