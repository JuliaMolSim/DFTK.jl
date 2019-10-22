# This script is called automatically upon package installation.
using PyCall
if PyCall.conda
    using Conda
    Conda.add("spglib"; channel="conda-forge")
    Conda.add("pymatgen"; channel="conda-forge")
end
