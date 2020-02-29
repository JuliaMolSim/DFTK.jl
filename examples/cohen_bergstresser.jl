using DFTK
import PyPlot
using DFTK.units: Ry

Ecut = 15               # kinetic energy cutoff in Hartree
n_bands_plot = 12       # number of bands to plot in the bandstructure
kline_density = 20      # Density of k-Points for bandstructure

# Setup silicon lattice
a = 10.263141334305942  # Silicon lattice constant in Bohr
Z = 14
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = Element(Z)
atoms = [Si => [ones(3)/8, -ones(3)/8]]


function potential_cohen_bergstresser(atomic_number, lattice_constant, G)
    # Get |G|^2 in units of (2π / lattice_constant)^2
    Gcart = basis.model.recip_lattice * G
    Gcartsq_pi = Int(round(sum(abs2, Gcart / (2π / lattice_constant)), digits=12))

    # We assume V_asym == 0 at the moment
    V_sym = Dict{Int64,Float64}()
    if atomic_number == 14      # Si
        V_sym[3]  = -0.21Ry
        V_sym[8]  =  0.04Ry
        V_sym[11] =  0.08Ry
    elseif atomic_number == 32  # Ge
        V_sym[3]  = -0.23Ry
        V_sym[8]  =  0.01Ry
        V_sym[11] =  0.06Ry
    elseif atomic_number == 50  # Sn
        V_sym[3]  = -0.20Ry
        V_sym[8]  =  0.00Ry
        V_sym[11] =  0.04Ry
    else
        error("Z == $atomic_number not implemented")
    end

    # The form factors in the Cohen-Bergstresser paper Table 2 are
    # with respect to non-normalised planewaves and are already
    # symmetrised into a sin-cos basis (see derivation p. 141)
    # => Scale by Ω / 2
    get(V_sym, Gcartsq_pi, 0.0) * basis.model.unit_cell_volume / 2
end


cb_potentials = [(G -> potential_cohen_bergstresser(Z, a, G)) => positions
                 for (at, positions) in atoms]
model = Model(lattice; atoms=atoms, external=term_external(cb_potentials))
basis = PlaneWaveBasis(model, Ecut)
ham = Hamiltonian(basis)


eReference = begin
    # Cohen-Bergstresser paper shows energies with reference
    # to the 3rd band at the Γ point
    n_reference = 3
    T = ComplexF64

    kpt = basis.kpoints[1]
    qrres = qr(randn(T, length(G_vectors(kpt)), n_reference + 3))
    res = lobpcg_hyper(kblock(ham, kpt), Matrix{T}(qrres.Q), tol=1e-5,
                      n_conv_check=n_reference)
    res.λ[n_reference]
end
plot_bands(ham, n_bands_plot, kline_density, eReference).show()
