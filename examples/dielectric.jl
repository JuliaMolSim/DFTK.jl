# Compute a few eigenvalues of the dielectric matrix (q=0,ω=0) iteratively

using DFTK
using Plots
using KrylovKit

# Calculation parameters
kgrid = [1, 1, 1]
Ecut = 5

# Silicon lattice
a = 10.26
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# change the symmetry to compute the dielectric operator with and without symmetries
model = model_LDA(lattice, atoms, symmetry=:off)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
scfres = self_consistent_field(basis, tol=1e-14)

# Apply ε = 1 - K χ0
function eps_fun(dv)
    dv = reshape(dv, size(scfres.ρ.real))
    dρ = apply_χ0(scfres.ham, dv, scfres.ψ, scfres.εF, scfres.eigenvalues)
    Kdρ = apply_kernel(basis, dρ; ρ=scfres.ρ)
    vec(dv - Kdρ)
end

# A straightfoward Arnoldi eigensolver that diagonalizes the matrix at each step
# This is more efficient than Arpack when `f` is very expensive
function arnoldi(f, x0, howmany; tol=1e-4, maxiter=30)
    for (V, B, r, nr, b) in ArnoldiIterator(f, x0)
        # A * V = V * B + r * b'
        V = hcat(V...)
        AV = V*B + r*b'

        ew, ev = eigen(B, sortby=real)
        Vr = V*ev
        AVr = AV*ev
        R = AVr - Vr * Diagonal(ew)

        # Select `howmany` smallest and largest eigenpairs
        N = size(V, 2)
        inds = 1:min(N, howmany)
        inds = [inds..., (min(N, howmany):N)...]

        normr = [norm(r) for r in eachcol(R[:, inds])]
        println(N)
        display([ew[inds] normr])
        if (N ≥ howmany && maximum(normr) < tol) || (N ≥ maxiter)
            return ew[inds], Vr[:, inds], AVr[:, inds]
        end
    end
end
arnoldi(eps_fun, vec(randn(size(scfres.ρ.real))), 5)
