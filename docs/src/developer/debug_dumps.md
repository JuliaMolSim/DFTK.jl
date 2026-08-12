# Debug dumping mechanism

Some failures of the code happen only rarely and are hard to reproduce. An example would
be issues in finding the right Fermi level for non-monotonous smearing functions. For
acquiring sufficient information when such issues occur and allow later debugging, DFTK
has a feature to automatically drop a json file with some context (e.g. a representation
of the basis, the eigenvalues and occupations, the SCF state etc.)
in case such failures happen.

To activate this mechanism you need to do two things:

1. Set the DFTK package-level preference `debugdump_prefix` to the location where you want
   the debugdump files to appear. This can be done conveniently using the 
   `DFTK.set_debugdump_prefix!("path/to/desired/location")` function.
   This you need to do exactly once.

2. In every script where you want to make use of debug dumping, make sure to
   add a `using JSON3` as the `JSON3` package (which is dynamically loaded) is needed
   for the debug dumping feature to work.
