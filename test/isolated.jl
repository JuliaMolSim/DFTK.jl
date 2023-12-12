@testitem "Isolated systems" tags=[:iso1, :off] begin
    using DFTK
    using LinearAlgebra

    RES = Dict()
    for a in (10, 20, 30)
        for per in (false, true)
            DFTK.reset_timer!(DFTK.timer)
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
            @show per, a
            display(DFTK.timer)
        end
    end
    display(sort(RES))
    @test true
end

# Broken while slab not implemented.
@testitem "Graphene surface" tags=[:iso2, :off] begin
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
        model_PBE(lattice, atoms, positions; periodic)
    end

    RES = Dict()
    for lz in (30, 50, 100, 150, 300)
        for per in (false, true)
            DFTK.reset_timer!(DFTK.timer)
            kgrid = [4, 4, 1]
            Ecut = 20
            tol = 1e-5

            model = model_bilayer(; lz, periodic=[true, true, per])
            basis = PlaneWaveBasis(model; Ecut, kgrid)
            res   = self_consistent_field(basis; tol)

            RES[per, lz] = res.energies.total
            @show per, lz
            display(DFTK.timer)
        end
    end
    display(sort(RES))
    @test true
end

@testitem "1d" tags=[:iso3, :off] begin
    using DFTK
    using LinearAlgebra
    using Plots
    Plots.scalefontsizes()  # reset
    Plots.scalefontsizes(2)
    pyplot()
    close()

    if false
        x = 1.234
        M = 2.345
        α = 3.456
        d = 1
        ε = 1e-4
        f(x) = DFTK.Vref_real(x, M, α, d)
        fpp(x) = -(f(x+ε)+f(x-ε)-2f(x))/(ε^2) / (4π)
        xs = range(0, 5, 100)
        p = plot(xs, fpp.(xs))
        plot!(p, xs, DFTK.ρref_real.(xs, M, α, d))
        display(p)
        println(-(f(x+ε)+f(x-ε)-2f(x))/(ε^2) - 4π*DFTK.ρref_real(x, M, α, d))
    end


    struct CustomPotential <: DFTK.Element
        α  # Prefactor
        L  # Width of the Gaussian nucleus

        α_charge
        L_charge
    end

    # Some default values
    CustomPotential() = CustomPotential(1.0, 0.5, 1.0, 0.5);

    function DFTK.local_potential_real(el::CustomPotential, r::Real)
        -el.α / (√(2π) * el.L) * exp(- (r / el.L)^2 / 2)
    end
    function DFTK.local_potential_fourier(el::CustomPotential, q::Real)
        -el.α * exp(- (q * el.L)^2 / 2)
    end

    function DFTK.charge_real(el::CustomPotential, r::Real)
        el.α_charge / (√(2π) * el.L_charge) * exp(- (r / el.L_charge)^2 / 2)
    end
    function DFTK.charge_fourier(el::CustomPotential, q::Real)
        el.α_charge * exp(- (q * el.L_charge)^2 / 2)
    end


    a = 10
    lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]];

    # x1 = 0.2
    # x2 = 0.8
    # positions = [[x1, 0, 0], [x2, 0, 0]]
    gauss     = CustomPotential(1, 0.5, 1, .5)
    positions = [[0.5, 0, 0]]
    atoms     = [gauss];


    gauss     = CustomPotential(1, 0.5, .333333, .5)
    δ = 2
    positions = [[1/2+δ/a, 0, 0],
                [1/2+δ/a, 0, 0],
                [1/2-δ/a, 0, 0]]
    atoms     = [gauss, gauss, gauss];

    # We setup a Gross-Pitaevskii model
    n_electrons = 1  # Increase this for fun
    terms = [Kinetic(),
             AtomicLocal(),
             Hartree(scaling_factor=.4),
             # LocalNonlinearity(ρ -> C * ρ^α)
            ]

    xlims = (-6, 6)
    p = plot(; xlims, reuse=false, legend=:bottomleft)
    for per in (true, false)
        periodic = fill(per, 3)
        model = Model(lattice, atoms, positions; n_electrons, terms,
                      spin_polarization=:spinless, periodic)

        # We discretize using a moderate Ecut and run a SCF algorithm to compute forces
        # afterwards. As there is no ionic charge associated to `gauss` we have to specify
        # a starting density and we choose to start from a zero density.
        basis = PlaneWaveBasis(model; Ecut=500, kgrid=(1, 1, 1))
        ρ = ones(eltype(basis), basis.fft_size..., 1)
        scfres = self_consistent_field(basis; tol=1e-8, ρ, maxiter=1000)

        ρ = scfres.ρ[:, 1, 1, 1]        # converged density, first spin component

        x = vec(first.(DFTK.r_vectors_cart(basis)))
        atomic_pot = (scfres.ham).blocks[1].operators[2].potential[:, 1, 1]; # use only dimension 1
        hartree_pot = (scfres.ham).blocks[1].operators[3].potential[:, 1, 1]; # use only dimension 1
        # plot(x, atomic_pot, label="Vat")
        dash = per ? :dash : :dot
        plot!(p, x.-a/2, ρ; ls=dash, label="ρ per=$per a=$a", xlims, lw=3)
        plot!(p, x.-a/2, atomic_pot; ls=dash, label="Vnucl per=$per a=$a", lw=3)
        plot!(p, x.-a/2, hartree_pot; ls=dash, label="Vh per=$per a=$a", lw=3)
    end
    display(p)
end

@testitem "1D toy system" tags=[:iso4, :off] begin
    using FFTW
    using Plots
    Plots.scalefontsizes()  # reset
    Plots.scalefontsizes(2)
    pyplot()  # for interactive plots
    close()

    N = 1000
    L = 10
    x = range(-L, L, N)
    ρ = @. exp(-(x) ^ 2) * (0.5 - (x)^2 + x)
    Vconv = map(1:N) do i
        -sum(abs(x[i] - x[j])*ρ[j] for j=1:N) * (2L/N) / 2
    end

    ks = map(1:N) do i
        K = i < N/2 ? i-1 : i-N - 1
        K * (2π/(2L))
    end

    poisson = @. 1 / (ks^2)
    poisson[1] = 0
    Vfft = real(ifft(poisson .* fft(ρ)))

    Vfft .-= Vfft[N÷2 + 1]
    Vconv .-= Vconv[N÷2 + 1]

    p = plot()
    p = plot!(p, x, ρ; label="ρ", lw=3, ls=:dot)
    p = plot!(p, x, Vconv; label="V_{ref}", lw=3)
    plot!(p, x, Vfft; label="V_{fft}", lw=3)

    # xlims = (-3,3)
    # ylims = (-5, 2)
    # plot!(p; xlims, ylims)

    display(p)
end

@testitem "1D ElementGaussian" tags=[:iso5, :off] begin
    using Plots
    using LinearAlgebra
    Plots.scalefontsizes()  # reset
    Plots.scalefontsizes(2)
    pyplot()  # for interactive plots
    close()
    p = plot()

    terms = [Kinetic(), AtomicLocal()]

    n_atoms = 1

    if n_atoms == 1
        X = ElementGaussian(1.0, 0.5, :X)
        atoms = [X]
        positions = [[0.0, 0.0, 0.0]]
    else
        X = ElementGaussian(1.0, 0.5, :X)
        Y = ElementGaussian(1.0, 0.2, :Y)
        atoms = [X, Y]
        positions = [[0.0, 0.0, 0.0], [1/4, 0, 0]]
    end
    n_atoms = length(atoms)
    lattice = diagm([5.0 * n_atoms, 0, 0])
    model = Model(lattice, atoms, positions; n_electrons=2, terms)
    basis = PlaneWaveBasis(model; Ecut=500, kgrid=[4,1,1])
    scfres = self_consistent_field(basis)
    ham = scfres.ham
    ρ = dropdims(scfres.ρ; dims=(2,3,4))

    potential = dropdims(DFTK.total_local_potential(ham); dims=(2,3,4))
    xs = [r[1] for r in dropdims(r_vectors(basis); dims=(2,3))]

    plot!(p, xs, ρ; label="ρ", lw=3, ls=:dot)
    plot!(p, xs, potential; label="V", lw=3)

    display(p)
end
