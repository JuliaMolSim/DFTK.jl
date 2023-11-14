@testitem "Isolated systems" begin
    using DFTK
    using LinearAlgebra

    RES = Dict()
    for a in (10, 20, 30)
        for per in (false, true)
            @time begin
                lattice = [a 0 0;
                        0 a 0;
                        0 0 a]
                atoms     = [ElementPsp(:Li, psp=load_psp("hgh/lda/Li-q3"))]
                atoms     = [ElementPsp(:He, psp=load_psp("hgh/lda/He-q2"))]
                positions = [[1/2, 1/2, 1/2]]

                kgrid = [1, 1, 1]  # no k-point sampling for an isolated system
                Ecut = 40
                tol = 1e-6

                model = model_LDA(lattice, atoms, positions, periodic=[per, per, per])
                basis = PlaneWaveBasis(model; Ecut, kgrid)
                res   = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(tol))

                rr = vec([norm(a * (r .- 1/2)) for r in r_vectors(basis)])

                function quadrupole(basis, ρ)
                    rr = [norm(a * (r .- 1/2)) for r in r_vectors(basis)]
                    sum(rr .^ 2 .* ρ) * basis.dvol
                end;
                quad = quadrupole(basis, res.ρ)
                println(quad)
                RES[per, a] = (quad, res.energies.total)
            end
        end
    end
    display(sort(RES))
    @test true
end

@testitem "Graphene surface" tags=[:graphene] begin
    using DFTK
    using LinearAlgebra

    model_bilayer(; lz=30, periodic=[true for _ in 1:3]) = let
        a = 4.66
        lattice = [  1/2     1/2  0;
                   -√3/2    √3/2  0;
                       0     0   lz/a] .* a
        positions = [[1/3, -1/3, -6.45/lz/2], [-1/3, 1/3, -6.45/lz/2],
                     [1/3, -1/3,  6.45/lz/2], [-1/3, 1/3,  6.45/lz/2]]
        atoms=fill(ElementPsp(6, psp=load_psp("hgh/pbe/c-q4")), 4)
        model_PBE(lattice, atoms, positions)
    end

    RES = Dict()
    for lz in (30, 50, 100, 150, 300)
        for per in (false, true)
            @time begin
                kgrid = [4, 4, 1]
                Ecut = 20
                tol = 1e-5

                model = model_bilayer(; lz, periodic=[true, true, per])
                basis = PlaneWaveBasis(model; Ecut, kgrid)
                res   = self_consistent_field(basis; tol)

                RES[per, lz] = res.energies.total
            end
        end
    end
    display(sort(RES))
    @test true
end
