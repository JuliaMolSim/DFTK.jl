# This script is called automatically upon package installation.
using PyCall
if PyCall.conda
    using Conda
    Conda.add("scipy")
    Conda.add("spglib"; channel="conda-forge")
end
