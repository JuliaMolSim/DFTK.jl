# Compute a few eigenvalues of the dielectric matrix (q=0,ω=0) iteratively

using DFTK
using KrylovKit

# Calculation parameters
kgrid = [1, 1, 1]
Ecut = 10

# Silicon lattice
a = 10.26
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# change the symmetry to compute the dielectric operator with and without symmetries
model = model_atomic(lattice, atoms; extra_terms=[Hartree()], symmetry=:off)
#  model = model_LDA(lattice, atoms; symmetry=:off)
fft_size = determine_grid_size(model, Ecut)
if all(iseven, fft_size)
    fft_size = fft_size .+ 1
end
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=fft_size)
scfres = self_consistent_field(basis, tol=1e-14)

# Apply ε = 1 - χ0 K
function eps_fun(dρ)
    dρ = reshape(dρ, size(scfres.ρ.real))
    Kdρ = apply_kernel(basis, from_real(basis, dρ); ρ=scfres.ρ)
    χ0Kdρ = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues, Kdρ)
    vec(dρ - χ0Kdρ.real)
end

# Apply ε'ϵ = (1 - K χ0) (1 - χ0 K)
function epseps_L2_fun(dρ)
    dρ = reshape(dρ, size(scfres.ρ.real))
    Kdρ = apply_kernel(basis, from_real(basis, dρ); ρ=scfres.ρ)
    χ0Kdρ = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues, Kdρ)
    dρ = dρ - χ0Kdρ.real
    χ0dρ = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues, from_real(basis, dρ))
    Kχ0dρ = apply_kernel(basis, χ0dρ; ρ=scfres.ρ)
    vec(dρ - Kχ0dρ.real)
end

# Apply ϵ_Vc = 1 - Vc^(1/2) χ0 K Vc^(-1/2)
function eps_Vc_fun(dρ)
    dρ = reshape(dρ, size(scfres.ρ.real))
    Vsdρ = DFTK.apply_kernel_invsqrt(basis.terms[6], from_real(basis, dρ))
    KVsdρ = apply_kernel(basis, Vsdρ; ρ=scfres.ρ)
    χ0KVsdρ = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues, KVsdρ)
    Vsχ0KVsdρ = DFTK.apply_kernel_sqrt(basis.terms[6], χ0KVsdρ)
    vec(dρ - Vsχ0KVsdρ.real)
end

# Apply ϵ_Vc'ϵ_Vc = (1 - Vc^(-1/2) K χ0 Vc^(1/2))(1 - Vc^(1/2) χ0 K Vc^(-1/2))
function epseps_Vc_fun(dρ)
    dρ = reshape(dρ, size(scfres.ρ.real))
    Vsdρ = DFTK.apply_kernel_invsqrt(basis.terms[6], from_real(basis, dρ))
    KVsdρ = apply_kernel(basis, Vsdρ; ρ=scfres.ρ)
    χ0KVsdρ = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues, KVsdρ)
    Vsχ0KVsdρ = DFTK.apply_kernel_sqrt(basis.terms[6], χ0KVsdρ)
    dρ = dρ - Vsχ0KVsdρ.real
    Vsdρ = DFTK.apply_kernel_sqrt(basis.terms[6], from_real(basis, dρ))
    χ0Vsdρ = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues, Vsdρ)
    Kχ0Vsdρ = apply_kernel(basis, χ0Vsdρ; ρ=scfres.ρ)
    VsKχ0Vsdρ = DFTK.apply_kernel_invsqrt(basis.terms[6], Kχ0Vsdρ)
    vec(dρ - VsKχ0Vsdρ.real)
end

# A straightfoward Arnoldi eigensolver that diagonalizes the matrix at each step
# This is more efficient than Arpack when `f` is very expensive
println("Starting Arnoldi ...")
function arnoldi(f, x0, howmany; tol=1e-8 , maxiter=30)
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
        inds = unique(append!(collect(1:min(N, howmany)), max(1, N-howmany):N))

        normr = [norm(r) for r in eachcol(R[:, inds])]
        println("#--- $N ---#")
        println("idcs      evals     residnorms")
        any(imag.(ew[inds]) .> 1e-5) && println("Warn: Suppressed imaginary part.")
        display(real.([inds ew[inds] normr]))
        println()
        if (N ≥ howmany && maximum(normr) < tol) || (N ≥ maxiter)
            return ew[inds], Vr[:, inds], AVr[:, inds]
        end
    end
end
howmany = 5
#  start = vec(randn(size(scfres.ρ.real)))
#  start[1] = 0
#  egval, _ = arnoldi(eps_fun, start, howmany)
#  singval_L2, _ = arnoldi(epseps_L2_fun, start, howmany)
#  egval_Vc, _ = arnoldi(eps_Vc_fun, start, howmany)
#  singval_Vc, _ = arnoldi(epseps_Vc_fun, start, howmany)
#  display([real.(egval) real.(sqrt.(singval_L2)) real.(egval_Vc) real.(sqrt.(singval_Vc))])

## for debugging with full matrices
#  χ0 = compute_χ0(scfres.ham)
#  Vc = DFTK.compute_kernel(basis.terms[6])
#  Vc_sqrt = DFTK.compute_kernel_sqrt(basis.terms[6])

#  ϵ = I - Vc * χ0
#  ϵ_Vc = I - Vc_sqrt * χ0 * Vc_sqrt
#  full_egval = [eigvals(ϵ) sqrt.(eigvals(ϵ'ϵ)) eigvals(ϵ_Vc) sqrt.(eigvals(ϵ_Vc'ϵ_Vc))]
#  display(real.([full_egval[1:5, :]; full_egval[end-5:end, :]]))
