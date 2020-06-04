# Compute the full dielectric matrix (q=0,ω=0)

using DFTK
using Plots
using KrylovKit

# Calculation parameters
kgrid = [1, 1, 1]       # k-Point grid
Ecut = 15 # energy cutoff in Hartree

# Setup silicon lattice
a = 10.263141334305942  # Silicon lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

supercell = [1, 1, 1]   # Lattice supercell
# Make a supercell if desired
pystruct = pymatgen_structure(lattice, atoms)
pystruct.make_supercell(supercell)
lattice = load_lattice(pystruct)
atoms = [Si => [s.frac_coords for s in pystruct.sites]]

model = model_LDA(lattice, atoms, symmetry=:off, temperature=0.001)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
@time scfres = self_consistent_field(basis, tol=1e-14, mixing=KerkerMixing())

# chi0 = compute_χ0(scfres.ham)
# Kh = DFTK.compute_hartree_kernel(basis)
# Kxc = DFTK.compute_xc_kernel(basis, scfres.ρ.real)

# eps = I - (Kh+Kxc)*chi0
# res = eigsolve(eps, length(scfres.ρ.real), 3, :SR)

function sym(v)
    DFTK.symmetrize(from_real(basis, v)).real
end

function epsfun(dv)
    dv = reshape(dv, size(scfres.ρ.real))
    dv_sym = dv
    dv_sym = sym(dv_sym) # comment this out to disable symetrization
    dρ = apply_χ0(scfres.ham, dv_sym, scfres.ψ, scfres.εF, scfres.eigenvalues)
    Kdρ = DFTK.apply_hartree_kernel(basis, dρ) + DFTK.apply_xc_kernel(basis, scfres.ρ.real, dρ)
    vec(dv - Kdρ)
end

function arnoldi(f, x0, howmany; tol=1e-4, maxiter=30)
    for (V, B, r, nr, b) in ArnoldiIterator(f, x0)
        # A * V = V * B + r * b'.
        V = hcat(V...)
        AV = V*B + r*b'

        ew, ev = eigen(B, sortby=real)
        Vr = V*ev
        AVr = AV*ev
        R = AVr - Vr * Diagonal(ew)

        N = size(V, 2)
        inds = 1:min(N, howmany)
        inds = [inds..., (inds .+ N .- min(N, howmany))...]
        normr = [norm(r) for r in eachcol(R[:, inds])]
        println(N)
        display([ew[inds] normr])
        if (N ≥ howmany && maximum(normr) < tol) || (N ≥ maxiter)
            return Vr[:, inds], AVr[:, inds]
        end
    end
end
arnoldi(epsfun, vec(randn(size(scfres.ρ.real))), 5)

# @time e1, v1 = eigsolve(epsfun, length(scfres.ρ.real), 3, :LR, verbosity=3, tol=1e-5, krylovdim=20)
# # e2, v2 = eigsolve(eps, length(scfres.ρ.real), 3, :SR)
# display(e1)
