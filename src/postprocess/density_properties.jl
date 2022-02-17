#
# Computational routines for basic density properties
# (charge
#


function print_density_properties(io::IO, basis::PlaneWaveBasis, ρ)
    magn  = sum(spin_density(ρ))  * basis.dvol
    nelec = sum(total_density(ρ)) * basis.dvol


# local density integrated around atoms

# magnetisation

# Partial charges

end



function print_citations(io::IO, basis::PlaneWaveBasis)
    xc = only(xc for xc in basis.terms if xc isa TermXc)



    xc.functionals[1].references



    xc.functionals[2].references


# Show citations (also DFT functional and psps)
# Show employed methods
end


function print_scf_summary(io::IO, scfres::NamedTuple)



    println(io, scfres.energies)



end
print_scf_summary(scfres::NamedTuple) = print_scf_summary(stdout, scfres)
