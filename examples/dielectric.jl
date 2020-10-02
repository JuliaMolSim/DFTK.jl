# Compute a few eigenvalues of the dielectric matrix (q=0,ω=0) iteratively

using DFTK
using Plots
using KrylovKit
using Printf

# Calculation parameters
kgrid = [1, 1, 1]
Ecut = 5

# Silicon lattice
a = 10.26
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# change the symmetry to compute the dielectric operator with and without symmetries
model = model_LDA(lattice, atoms, symmetry=false)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
scfres = self_consistent_field(basis, tol=1e-14)

# Apply ε† = 1 - χ0 (vc + fxc)
function eps_fun(dρ)
    dρ = reshape(dρ, size(scfres.ρ.real))
    dρ = from_real(basis, dρ)
    dv = apply_kernel(basis, dρ; ρ=scfres.ρ)
    χdv = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues, dv)
    vec((dρ - χdv).real)
end

# A straightfoward Arnoldi eigensolver that diagonalizes the matrix at each step
# This is more efficient than Arpack when `f` is very expensive
println("Starting Arnoldi ...")
function arnoldi(f, x0; howmany=5, tol=1e-4, maxiter=30, n_print=howmany)
    for (V, B, r, nr, b) in ArnoldiIterator(f, x0)
        # A * V = V * B + r * b'
        V = hcat(V...)
        AV = V*B + r*b'

        ew, ev = eigen(B, sortby=real)
        Vr = V*ev
        AVr = AV*ev
        R = AVr - Vr * Diagonal(ew)

        N = size(V, 2)
        normr = [norm(r) for r in eachcol(R)]

        println("#--- $N ---#")
        println("idcs      evals     residnorms")
        inds = unique(append!(collect(1:min(N, n_print)), max(1, N-n_print):N))
        for i in inds
            @printf "% 3i  %10.6g  %10.6g\n" i real(ew[i]) normr[i]
        end
        any(imag.(ew[inds]) .> 1e-5) && println("Warn: Suppressed imaginary part.")
        println()

        is_converged = (N ≥ howmany && all(normr[1:howmany] .< tol)
                        && all(normr[end-howmany:end] .< tol))
        if is_converged || (N ≥ maxiter)
            return (λ=ew, X=Vr, AX=AVr, residual_norms=normr)
        end
    end
end
arnoldi(eps_fun, vec(randn(size(scfres.ρ.real))); howmany=5)
