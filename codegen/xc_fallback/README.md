This directory provides routines that are used to generate the code in
`src/core/xc_fallback`. The intention is to be able to quickly add a
pure-Julia fallback implementation for simple functionals, such that
calculations in deviating precision or using non-standard Array types
can be performed. For the standard case DFTK still relies on libxc via Libxc.jl.

The generators in this folder are based upon the Maple code generators
used by [libxc](https://tddft.org/programs/libxc) in order to
implement its exchange-correlation functionals. Currently the
generation procedure only generates rough code, which needs to be
manually edited in order to compile and to work. Thus a proper
mass-generation of functionals would require a more sophisticated
solution, but the implementation of selected important cases should be
possible like so.
