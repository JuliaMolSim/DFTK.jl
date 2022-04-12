@testitem "Isolated systems" tags=[:off] begin
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
end
