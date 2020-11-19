using DFTK
using ProgressMeter
using LinearAlgebra

include("../dftk_to_wannier90.jl")
include("../win_guess_utils.jl")

a = 10.26 #a.u.

lattice = a / 2*[[-1.  0. -1.];   #basis.model.lattice (in a.u.)
                 [ 0   1.  1.];
                 [ 1   1.  0.]]

Si = ElementPsp(:Si, psp=load_psp("hgh/pbe/Si-q4"))
atoms = [ Si => [zeros(3), 0.25*[-1,3,-1]] ]
model = model_PBE(lattice,atoms)

kgrid = [4,4,4] # mp grid
Ecut = 20
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, optimize_fft_size = true, use_symmetry = false)

scfres = self_consistent_field(basis, tol=1e-12, n_bands = 12, n_ep_extra = 0 );
ψ = scfres.ψ



########################################################
#                                                      #
##### ### #             TESTS                # ### #####
#                                                      #
########################################################

# Here, bands at neighbours k=3 and k=62 are sperated
# The svd decompositions of matrices given by QE and Julia should be equal.

# function change_format_mmn(M,n_bands)
#     M_gf  = [a + b*im for (a,b) in [M[i,:] for i in 1:n_bands*n_bands] ]
#     M_gf = reshape(M_gf,(n_bands,n_bands))
#     M_gf
# end



# # Tests on several atoms. Random system in order to test routines on the basis.
# Si = ElementPsp(:Si, psp=load_psp("hgh/pbe/Si-q4"))
# Mg = ElementPsp(:Mg, psp=load_psp("hgh/pbe/Mg-q2"))
# atoms = [ Si => [zeros(3), 0.25*[-1,3,-1]]
#           Mg => [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]]
#           ]



# # Centers of the gaussian guesses for silicon. Test for subroutines
# Si_centers = [[-0.125,-0.125, 0.375], [0.375,-0.125,-0.125], [-0.125, 0.375,-0.125], [-0.125,-0.125,-0.125]]
# test_proj =  [ [-0.12500,0.12500,-0.1250],[0,1,1],[0.0000,0.0000,1.0000],[1.00000,0.00000,0.00000],1.00 ]
