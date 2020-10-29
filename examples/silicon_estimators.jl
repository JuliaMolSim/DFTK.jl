# Very basic setup, useful for testing
using DFTK
using LinearAlgebra
import KrylovKit: ArnoldiIterator, Orthogonalizer, OrthonormalBasis, KrylovDefaults, orthogonalize!
using KrylovKit

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_LDA(lattice, atoms, n_electrons=8)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 15           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

scfres = self_consistent_field(basis, tol=1e-10)
display(scfres.energies)

tol_test = 1e-12

# for a given kpoint, we compute the
# projection of a vector ϕk on the orthogonal of the eigenvectors ψk
function proj!(ϕk, ψk, occk)

    N = length([l for l in occk if l != 0.0])
    Πϕk = deepcopy(ϕk)
    for i = 1:N, j = 1:N
        Πϕk[:,i] -= (ψk[:,j]'ϕk[:,i]) * ψk[:,j]
    end
    for i = 1:N, j = 1:N
        @assert abs(Πϕk[:,i]'ψk[:,j]) < tol_test [println(abs(Πϕk[:,i]'ψk[:,j]))]
    end
    ϕk = Πϕk
    ϕk
end

# KrylovKit custom orthogonaliser
struct OrthogonalizeAndProject{F, O <: Orthogonalizer, ψk, occk} <: Orthogonalizer
    projector!::F
    orth::O
    ψ::ψk
    occ::occk
end
OrthogonalizeAndProject(projector, ψk, occk) = OrthogonalizeAndProject(projector,
                                                                       KrylovDefaults.orth,
                                                                       ψk, occk)

function KrylovKit.orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector,
                                        alg::OrthogonalizeAndProject) where {T}
    v, x = orthogonalize!(v, b, x, alg.orth)
    v = reshape(v, size(alg.ψ))
    v = vec(alg.projector!(v, alg.ψ, alg.occ))::T
    v, x
end
#
# generate random δφ that are all orthogonal to every ψi for 1 <= i <= N
function generate_δφ(ψk, occk)

    N = length([l for l in occk if l != 0.0])

    # generate random vector and project it
    δφk = rand(typeof(ψk[1,1]), size(ψk))
    δφk = proj!(δφk, ψk, occk)

    # normalization and test
    for i = 1:N
        δφk[:,i] /= norm(δφk[:,i])
        for j = 1:N
            @assert abs(δφk[:,i]'ψk[:,j]) < tol_test [println(abs(δφk[:,i]'ψk[:,j]))]
        end
    end
    δφk
end
#
# for a given kpoint, we compute the
# application of Ω to an element on the tangent plane
# Here, an element on the tangent plane can be written as
#       δP = Σ |ψi><δφi| + hc
# where the δφi are of size Nb and are all orthogonal to the ψj, 1 <= j <= N
# therefore we store them in the same kind of array than ψ, with
# δφ[ik][:,i] = δφi for each k-point
# therefore, computing Ωδφ can be done analitically
function apply_Ω(δφk, ψk, H::HamiltonianBlock, occk, egvalk)

    basis = scfres.basis

    N = length([l for l in occk if l != 0.0])

    Ωδφk = 0. * copy(δφk)

    Hδφk = H * δφk
    Hδφk = proj!(Hδφk, ψk, occk)

    # compute component on i
    for i = 1:N
        ε_i = egvalk[i]
        Ωδφk[:,i] = Hδφk[:,i] - ε_i * δφk[:,i]
    end
    Ωδφk
end

#
function apply_K()
end




# Compare eigenvalues of Ω with the gap
function validate_Ω(scfres)

    ψ = scfres.ψ
    basis = scfres.basis
    occ = scfres.occupation
    egval = scfres.eigenvalues
    H = scfres.ham
    vecs = []
    vals = []
    gap = nothing

    for ik = 1:length(basis.kpoints)

        occk = occ[ik]
        egvalk = egval[ik]
        N = length([l for l in occk if l != 0.0])
        gap = egvalk[N+1] - egvalk[N]

        ψk = ψ[ik][:,1:N]
        Hk = H.blocks[ik]
        δφk = generate_δφ(ψk, occk)
        x0 = vec(δφk)

        # function we want to compute eigenvalues
        function f(x)
            ncalls += 1
            x = reshape(x, size(δφk))
            x = proj!(x, ψk, occk)
            Ωx = apply_Ω(x, ψk, Hk, occk, egvalk)
            Ωx = proj!(Ωx, ψk, occk)
            vec(Ωx)
        end

        println("\n--------------------------------")
        println("Solving with KrylovKit...")
        ncalls = 0
        vals_Ω, vecs_Ω, info = eigsolve(f, x0, 8, :SR;
                                        tol=1e-6, verbosity=1, eager=true,
                                        maxiter=50, krylovdim=30,
                                        orth=OrthogonalizeAndProject(proj!, ψk, occk))

        println("\nKryloKit calls to operator: ", ncalls)
        idx = findfirst(x -> abs(x) > 1e-6, vals_Ω)
        display(vals_Ω)
        println("\n")
        display(info.normres)
        println("\n")
        display(vals_Ω[idx])
        display(gap)
        display(norm(gap-vals_Ω[idx]))
        append!(vecs, vecs_Ω)
        append!(vals, vals_Ω)
    end
    vals, vecs, gap
end

vals, vecs, gap = validate_Ω(scfres)
