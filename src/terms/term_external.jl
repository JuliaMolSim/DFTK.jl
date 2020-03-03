# Functions returning appropriate builders for the external potential

"""
    build_local_potential(pw::PlaneWaveBasis, generators_or_atoms...;
                          compensating_background=true)

Function generating a local potential on the real-space density grid ``B^∗_ρ``
defined by the plane-wave basis `pw`. The potential is generated by summing
(in Fourier space) analytic contributions from all species involved in the
lattice, followed by an iFFT. The lattice definition is taken implicitly from `pw`.

The contributions are defined by the `generators_or_atoms` pairs. In the simplest case
these are pairs from `Element` objects to lists of fractional coordinates defining the
real-space positions of the atoms. More generally any function `G -> potential(G)`, which
evaluates a local potential at this reciprocal space position may be used. In this
case `G` is passed in integer coordinates.  
The parameter `compensating_background` (default true) determines whether the DC component
will be automatically set to zero, which physically corresponds to including
a compensating change background in the potential model.

# Examples
Given an appropriate lattice and basis definition in `basis` one may build
the local potential for an all-electron treatment of sodium chloride as such
```julia-repl
julia> na = Element(11); cl = Element(17)
       build_local_potential(basis, [na => [[0,0,0], [1/2,1/2,0], [1/2,0,1/2], [0,1/2,1/2]],
                                     cl => [[0,1/2,0], [1/2,0,0], [0,0,1/2], [1/2,1/2,1/2]]])
```
Equivalently one could have explicitly specified the Coulomb potential function to
be used, e.g.
```julia-repl
julia> na_Coulomb(G) = -11 / sum(abs2, basis.recip_lattice * G)
       cl = Element(17)
       build_local_potential(basis,
                             [na_Coulomb => [[0,0,0], [1/2,1/2,0], [1/2,0,1/2], [0,1/2,1/2]],
                              cl => [[0,1/2,0], [1/2,0,0], [0,0,1/2], [1/2,1/2,1/2]]])
```
since sodium has nuclear charge 11.
```
"""
function term_external(atoms)
    function inner(basis::PlaneWaveBasis{T}, energy, potential; ρ=nothing, kwargs...) where T
        model = basis.model

        # gen(G) = int_Ω Vper(x) e^-iGx
        #        = int_R^3 V(x) e^-iGx
        #        = Ω <e_G, Vper e_0>
        make_generator(elem::Function) = elem
        function make_generator(elem::Element)
            return G -> local_potential_fourier(sum(abs2, model.recip_lattice * G))
        end
        genfunctions = [make_generator(elem) => positions
                        for (elem, positions) in generators_or_atoms]

        # We expand Vper in the basis set:
        # Vper(r) = sum_G cG e_G(r)
        # cG = <e_G, Vper> = gen(G) / sqrt(Ω)
        coeffs = map(G_vectors(basis)) do G
            sum(Complex{T}(
                1/sqrt(model.unit_cell_volume)
                * genfunction(G)          # Potential data for wave vector G
                * cis(-2T(π) * dot(G, r))     # Structure factor
                ) for (genfunction, positions) in genfunctions
                  for r in positions
           )
        end

        # TODO impose Vext to be real
        Vext = (potential === nothing) ? G_to_r(basis, coeffs) : G_to_r!(potential, basis, coeffs)
        if energy !== nothing
            dVol = model.unit_cell_volume / prod(basis.fft_size)
            energy[] = real(sum(ρ.real .* Vext)) * dVol
            @assert_expensive energy[] ≈ real(dot(ρ.fourier, coeffs))
        end

        energy, potential
    end
    inner
end
