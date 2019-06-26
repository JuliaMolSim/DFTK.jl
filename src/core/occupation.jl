"""
Compute the occupation at zero temperature and without smearing
for `n_elec` electrons and the bands `Psi` with associated `energies`.
"""
function occupation_zero_temperature(basis, energies, Psi, n_elec)
    n_bands = size(Psi[1], 2)

    @assert n_elec % 2 == 0 "Only even number of electrons implemented"
    @assert n_bands ≥ n_elec / 2

    T = eltype(basis.lattice)
    occupation = similar(basis.kpoints, Vector{T})
    for ik in 1:length(occupation)
        occupation[ik] = zeros(T, n_bands)
        occupation[ik][1:Int(n_elec / 2)] .= 2
    end
    occupation
end

# TODO Implement smearing
