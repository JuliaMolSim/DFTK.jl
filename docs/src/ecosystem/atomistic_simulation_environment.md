# Atomistic simulation environment

The [atomistic simulation environment](https://wiki.fysik.dtu.dk/ase/index.html),
or ASE for short,
is a popular Python package to simplify the process of setting up,
running and analysing results from atomistic simulations across different simulation codes.
By means of [ASEconvert](https://github.com/mfherbst/ASEconvert.jl) this python package
is seamlessly integrated with the AtomsBase ecosystem and thus available to DFTK via
our own [AtomsBase integration](@ref) and [AtomsCalculators integration](@ref).

Most notably this integration is two-way, that is that DFTK can both be used as a
calculator in ASE (to do computation on atomistic systems managed in python) and vice
versa, i.e. atomistic systems set up in ASE can be used as starting points for
calculations in a Julia-driven workflow. Both will be illustrated.

## Using ASE to setup structures for DFTK calculations

An example of this workflow is given in [Modelling a gallium arsenide surface](@ref).

## Using DFTK as a calculator in ASE

!!! warning "Experimental feature"
    This feature is basically untested and at the moment mainly serves as a
    pointer to the expert user how a DFTK integration into ASE could be achieved.
    If you are interested in trying this and ironing out the rough edges,
    please open an issue or a PR to polish this towards a state where it can
    be helpful for others.

The [AtomsCalculatorsUtilities](https://github.com/JuliaMolSim/AtomsCalculatorsUtilities.jl)
package contains an implementation of the [i-PI protocol](https://github.com/i-pi/i-pi)
to pass energies, forces and virials between different atomistic calculators.
This can be used, for example, to make julia-based calculators available in ASE.
Note that at the moment AtomsCalculatorsUtilities is not yet registered
in general and you will need to install it from the github url.

On the python-side (for ASE), this requests the computation of a force
of a hydrogen molecule:

```python
from ase import Atoms
from ase.calculators.socketio import SocketIOCalculator

h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]])

calc = SocketIOCalculator(log="test.log")

h2.calc = calc
h2.get_forces()  # Stalls until you connect a driver
```

To supply this request with an LDA force based on DFTK,
we need to setup an initial system as a template
as well as an LDA calculator with desired numerical parameters:

```julia
using AtomsBase
using AtomsCalculatorsUtilities.IPI
using DFTK
using Unitful

# Setup initial system
bounding_box = [[10.0, 0, 0],
                [0, 10.0, 0],
                [0, 0, 10.0]]u"Å"
hydrogen = periodic_system([
    :H => [0, 0, 0.]u"Å",
    :H => [0, 0, 1.]u"Å"
], bounding_box)

# Setup DFTK calculator
calc = DFTKCalculator(;
    model_kwargs = (; functionals=LDA()),
    basis_kwargs = (; kgrid=[1, 1, 1], Ecut=10)
)

# Run IPI driver
run_driver("127.0.0.1", calc, hydrogen)
```
