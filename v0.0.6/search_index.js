var documenterSearchIndex = {"docs":
[{"location":"#DFTK.jl:-The-density-functional-toolkit.-1","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"","category":"section"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"DFTK is a julia package of for playing with plane-wave density-functional theory algorithms.","category":"page"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"TODO the notations in this file are out of date. Look at the examples or at the source code for updated documentation.","category":"page"},{"location":"#Terminology-and-Definitions-1","page":"DFTK.jl: The density-functional toolkit.","title":"Terminology and Definitions","text":"","category":"section"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"The general terminology used throughout the documentation of the plane-wave aspects of the code.","category":"page"},{"location":"#Lattices-1","page":"DFTK.jl: The density-functional toolkit.","title":"Lattices","text":"","category":"section"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"Usually we denote with A the matrix containing all lattice vectors as columns and with","category":"page"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"textbfB = 2pi textbfA^-T","category":"page"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"the matrix containing the reciprocal lattice vectors as columns.","category":"page"},{"location":"#Units-1","page":"DFTK.jl: The density-functional toolkit.","title":"Units","text":"","category":"section"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"Unless otherwise stated the code and documentation uses atomic units and fractional or integer coordinates for k-Points and wave vectors G. The equivalent Vectors in cartesian coordiates will be denoted as k^c or G^c, i.e.","category":"page"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"k^c = textbfB k quad G^c = textbfB G","category":"page"},{"location":"#Plane-wave-basis-functions-1","page":"DFTK.jl: The density-functional toolkit.","title":"Plane wave basis functions","text":"","category":"section"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"At the moment the code works exclusively with orthonormal plane waves. In other words our bases consist of functions","category":"page"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"e_G^c = 1sqrtOmega e^i G^c cdot x","category":"page"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"where Omega is the unit cell volume and G^c is a wave vector in cartesian coordiates.","category":"page"},{"location":"#Basis-sets-1","page":"DFTK.jl: The density-functional toolkit.","title":"Basis sets","text":"","category":"section"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"The wave-function basis B_k, consisting of all plane-wave basis functions below the desired energy cutoff E_textcut for each k-point:\nB_k =  e_G^c  12 G^c + k^c^2  E_textcut\nGeometrically the corresponding wave vectors G^c form a ball of radius sqrt2 E_textcut centred at k^c. This makes the corresponding set of G-vectors\n G  textbfB (G + k)  2 sqrtE_textcut \nin integer coordinates an ellipsoid.\nThe potential or density basis B_rho, consisting of all plane waves on which a potential needs to be known in order to be consistent with the union of all B_k for all k. This means that it is the set\nB_rho =  e_G^c - e_tildeG^c  e_G^c e_tildeG^c in B_k \nThis is equivalent to the alternative definition\nB_rho =  e_G^c  12 G^c^2  α^2 E_textcut \nfor a supersampling factor alpha = 2. Geometrically this is again a ball in cartesian coordinates and an ellipsoid in integer coordinates.\nIn practice we do not use B_rho in the code, since fast-fourier transforms (FFT) operate on rectangular grids instead. For this reason the code determines C_rho, the smallest rectangular grid in integer coordinates which contains all G-vectors corresponding to the plane waves of B_rho. For this we take\nC_rho =  G = (G_1 G_2 G_3)^T  G_i  N_i \nwhere the bounds N_i are determined as follows. Since G = textbfB^-1 G^c one can employ Cauchy-Schwartz to get\nN_i = max_G^c^2  2 α^2 E_textcut(textbfB^-1i  cdot G^c)\n     textbfB^-1i  sqrt2 α^2 E_textcut\nWith textbfB^-1 = frac12pi textbfA^T therefore\nN_i  textbfA i fracsqrt2 α^2 E_textcut2π\nwhere e.g. textbfA i denotes the i-th column of textbfA. Notice, that this makes C_rho is a rectangular shape in integer coordinates, but a parallelepiped in cartesian coordinates.","category":"page"},{"location":"#TODO-not-yet-properly-updated-from-here-1","page":"DFTK.jl: The density-functional toolkit.","title":"TODO not yet properly updated from here","text":"","category":"section"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"The XC basis B_textXC, which is used for computing the application of the exchange-correlation potential operator to the density rho, represented in the basis B_rho, that is\nB_textXC  = e_G  12 G_textmax^2  β E_textcut \nSince the exchange-correlation potential might involve arbitrary powers of the density ρ, a numerically exact computation of the integral\nlangle e_G  V_textXC(ρ) e_G rangle qquad textwith qquad e_G e_G  B_Ψk\nrequires the exchange-correlation supersampling factor beta to be infinite. In practice, beta =4 is usually chosen, such that B_textXC = B_rho.","category":"page"},{"location":"#Real-space-grids-1","page":"DFTK.jl: The density-functional toolkit.","title":"Real-space grids","text":"","category":"section"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"Due to the Fourier-duality of reciprocal-space and real-space lattice, the above basis sets define corresponding real-space grids as well:","category":"page"},{"location":"#","page":"DFTK.jl: The density-functional toolkit.","title":"DFTK.jl: The density-functional toolkit.","text":"The grid B_rho^ast, the potential integration grid, which is the grid used for convolutions of a potential with the discretized representation of a DFT orbital. It is simply the iFFT-dual real-space grid of B_rho.\nThe grid B^ast_textXC, the exchange-correlation integration grid, i.e. the grid used for convolutions of the exchange-correlation functional terms with the density or derivatives of it. It is the iFFT-dual of B_textXC.","category":"page"}]
}
