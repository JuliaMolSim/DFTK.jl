# Unit test system

This document gives some details how to run and develop unit tests for DFTK.

We use [TestItemRunner](https://github.com/julia-vscode/TestItemRunner.jl) to manage the
tests. It reduces the risk to have undefined behavior by preventing tests from being run in
global scope.
Moreover, it allows for greater flexibility by providing ways to launch a specific subset of
the tests. 

## Running tests by tags
To only run a minimal set of tests designed to ensure DFTK functionality (tests tagged with `:minimal`),
one can simply run
```julia
using Pkg
Pkg.test("DFTK"; test_args = ["minimal"])
```
If no `test_args` are given, all tests are run. Specifying any subset implicitly turns off all tests not
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

## Test-driven development
Oftentimes you want to iterate on either a test, or the corresponding code being tested.
Running `Pkg.test` will instantiate a new test environment, precompile DFTK, etc... every single time.
It should thus be avoided for quick iteration.

Instead, a workflow that works well is the following:
1. Ensure that you have Revise.jl and TestEnv.jl installed in your default environment.
1. Start the REPL in the DFTK directory.
1. Run `using TestEnv, Revise` (if they are not already in your `startup.jl` file).
1. Setup an environment with the DFTK test dependencies: `TestEnv.activate()`.
1. Run a specific test by name using TestItemRunner, for example:
   ```jl
   using TestItemRunner
   TestItemRunner.run_tests("test/", filter=ti->ti.name=="Hamiltonian consistency")
   ```
   This runs the test item with the name `Hamiltonian consistency`, i.e. declared in code as
   `@testitem "Hamiltonian consistency" ...`.
1. Modify either DFTK or test code, and call `run_tests` again as many times as necessary.
   Revise will ensure that changes to DFTK will be picked up.

### Other filters
TestItemRunner also supports selection by a file name.
For example, to run all tests in a unit test file named `serialisation.jl`:
```julia
TestItemRunner.run_tests("test/", filter=ti->occursin("serialisation.jl", ti.filename))
```

As the above syntax suggests filters can be more general,
using the `ti.filename`, `ti.name` and/or `ti.tags` fields passed to the filter.

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
