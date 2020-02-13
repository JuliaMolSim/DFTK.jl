# This script is called automatically upon package installation.
using PyCall
if PyCall.conda
    import Conda
    Conda.add("spglib==1.12.2"; channel="conda-forge")
    Conda.add("pymatgen"; channel="conda-forge")
end
