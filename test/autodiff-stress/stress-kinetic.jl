# Very basic setup, useful for testing
using DFTK
using LinearAlgebra
using BenchmarkTools

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_LDA(lattice, atoms)
kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 15          # kinetic energy cutoff in Hartree -- can increase to make G_vectors larger (larger solve time)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

@time scfres = self_consistent_field(basis, tol=1e-8) # 75.068789 seconds (138.55 M allocations: 8.145 GiB, 4.59% gc time, 24.68% compilation time)

# TODO try to rewrite for Zygote (performance optimizations)
# e.g. translate loops to dense arrays or maps (?)

function kinetic_energy(lattice, basis, ψ, occ)
    recip_lattice = 2π * inv(lattice')
    E = zero(Float64)
    kinetic_energies = [[sum(abs2, recip_lattice * (G + kpt.coordinate)) / 2
                         for G in  G_vectors(kpt)]
                        for kpt in basis.kpoints]
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[1], 2)
            ψnk = @views ψ[ik][:, iband]
            E += (basis.kweights[ik] * occ[ik][iband]
                  * real(dot(ψnk, kinetic_energies[ik] .* ψnk)))
        end
    end
    E
end
kinetic_energy(lattice) = kinetic_energy(lattice, basis, scfres.ψ, scfres.occupation)

@time E = kinetic_energy(lattice) # 0.438027 seconds (623.88 k allocations: 36.457 MiB, 99.96% compilation time)
@btime kinetic_energy(lattice) # 49.123 μs (742 allocations: 169.05 KiB)

# stress := diff E wrt lattice

#===#
# Check results and compile times on first call
stresses = Dict()

# works fine
using ForwardDiff
@time stresses[:ForwardDiff] = ForwardDiff.gradient(kinetic_energy, lattice) # 3.627630 seconds (5.99 M allocations: 363.981 MiB, 5.08% gc time, 98.69% compilation time)

# works but long compile time and gives ComplexF64 results
# hypothesis: slow compilation due to loops (and generators)
using Zygote
@time stresses[:Zygote] = Zygote.gradient(kinetic_energy, lattice) # 61.094425 seconds (63.31 M allocations: 3.715 GiB, 3.85% gc time, 67.43% compilation time)

# works fine
using ReverseDiff
@time stresses[:ReverseDiff] = ReverseDiff.gradient(kinetic_energy, lattice) # 5.409118 seconds (9.60 M allocations: 516.091 MiB, 14.61% gc time, 89.56% compilation time)

# sanity check
using FiniteDiff
@time stresses[:FiniteDiff] = FiniteDiff.finite_difference_gradient(kinetic_energy, lattice) # 2.606210 seconds (2.87 M allocations: 232.911 MiB, 19.92% gc time, 99.19% compilation time)

stresses
# Dict{Any, Any} with 4 entries:
# :ForwardDiff => [0.27005 -0.27005 -0.27005; -0.27005 0.27005 -0.27005; -0.27005 -0.27005 0.27005]
# :FiniteDiff  => [0.27005 -0.27005 -0.27005; -0.27005 0.27005 -0.27005; -0.27005 -0.27005 0.27005]
# :Zygote      => (ComplexF64[0.27005-0.0im -0.27005-0.0im -0.27005-0.0im; -0.27005-0.0im 0.27005-0.0im -0.27005-0.0im; -0.27005-0.0im -0.27005-0.0im 0.27005-0.0im],)
# :ReverseDiff => [0.27005 -0.27005 -0.27005; -0.27005 0.27005 -0.27005; -0.27005 -0.27005 0.27005]

@btime ForwardDiff.gradient(kinetic_energy, lattice) #    270.426 μs (   761 allocations:  1.07 MiB)
@btime Zygote.gradient(kinetic_energy, lattice)      #  6.983 ms     ( 34765 allocations: 12.61 MiB)
@btime ReverseDiff.gradient(kinetic_energy, lattice) # 15.376 ms     (415886 allocations: 16.42 MiB)
@btime FiniteDiff.finite_difference_gradient(kinetic_energy, lattice) # 777.578 μs (13394 allocations: 2.97 MiB)
