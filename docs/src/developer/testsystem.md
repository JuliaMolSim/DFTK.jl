# Unit test system

This document gives some details how to run and develop unit tests for DFTK.

We use [TestItemRunner](https://github.com/julia-vscode/TestItemRunner.jl) to manage the
tests. It reduces the risk to have undefined behavior by preventing tests from being run in
global scope.
Moreover, it allows for greater flexibility by providing ways to launch a specific subset of
the tests. 

## Running selective tests
### Selecting by tags
To only run a minimal set of tests designed to ensure DFTK functionality (tests tagged with `:minimal`),
on can simply run
```julia
using Pkg
Pkg.test("DFTK"; test_args = ["minimal"])
```
By default, all tests are run. Specifying any subset implicitly turns off all tests not
tagged accordingly. Multiple tags can be specified at once. For example,
```julia
using Pkg
Pkg.test("DFTK"; test_args = ["forces", "example"])
```
will test forces and run the examples. It is also possible to disable certain tests:
```julia
using Pkg
Pkg.test("DFTK"; test_args = ["noslow"])
```
will ignore any test tagged as `:slow`. Finally, parallel tests can be run by passing `"mpi"` to
the `test_args` keyword argument. GPU tests are triggered with the `"gpu"` tag.

### Selecting by file name
This works by directly instantiating the test environment and triggering
the `@run_package_tests` macro from `TestItemRunner` manually. For instance:
```julia
using TestEnv       # Optional: automatically installs required packages
TestEnv.activate()  # for tests in a temporary environment.
using TestItemRunner
using DFTK
cd(joinpath(pkgdir(DFTK), "test"))
@run_package_tests filter = ti -> occursin("serialisation.jl", ti.filename)
```
would only run the tests of the particular unit test file `serialisation.jl`.

### More general filters
As the above syntax suggests filters can be more general,
see the [TestItemRunner documentation](https://github.com/julia-vscode/TestItemRunner.jl/#running-tests)
for more details.

## Developing unit tests
If you need to write tests, note that you can create modules with `@testsetup`. To use
a function `my_function` of a module `MySetup` in a `@testitem`, you can import it with
```julia
using .MySetup: my_function
```
It is also possible to use functions from another module within a module. But for this the
order of the modules in the `setup` keyword of `@testitem` is important: you have to add the
module that will be used before the module using it. From the latter, you can then use it
with
```julia
using ..MySetup: my_function
```
