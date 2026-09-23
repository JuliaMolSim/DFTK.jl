This folder contains the necessary logic for GPU CI/CD on CSCS infrastructure.

There is a short list of GitHub users whose actions that will trigger the CI/CD pipeline. These actions are:
- opening a PR against master
- writing a comment in a PR with `cscs-ci run Mi300`, `cscs-ci run GH200` or `cscs-ci run GH200, Mi300`

And the privileged users are:
- abussy
- mfherbst
- antoine-levitt

The CI/CD pipelines will be run on CSCS infrastructure for free. Depending on the state of the machines 
(maintenance, shut-down, etc.), the CI might not run. In this case, the tests can be re-triggered by
a comment from a privileged user, i.e. `cscs-ci run GH200, Mi300`.

If there is a problem, feel free to raise an issue, or to contact abussy (augustin.bussy@cscs.ch) directly.
