# Useful formulas

This section holds a collection of formulae, which are helpful
when working with DFTK and plane-wave DFT in general.
See also [Notation and conventions](@ref) for a description
of the conventions used in the equations.

## Fourier transforms
- The Fourier transform is
  ```math
  \widehat{f}(q) = \int_{{\mathbb R}^{3}} e^{-i q \cdot x} f(x) dx
  ```
- Fourier transforms of centered functions: If ``f({x}) = R(x) Y_l^m(x/|x|)``, then
  ```math
  \begin{aligned}
    \hat f( q)
    &= \int_{{\mathbb R}^3} R(x) Y_{l}^{m}(x/|x|) e^{-i {q} \cdot {x}} d{x} \\
    &= \sum_{l = 0}^\infty 4 \pi i^l
    \sum_{m = -l}^l \int_{{\mathbb R}^3}
    R(x) j_{l'}(|q| |x|)Y_{l'}^{m'}(-q/|q|) Y_{l}^{m}(x/|x|)
     Y_{l'}^{m'\ast}(x/|x|)
    d{x} \\
    &= 4 \pi Y_{l}^{m}(-q/|q|) i^{l}
    \int_{{\mathbb R}^+} r^2 R(r) \ j_{l}(|q| r) dr\\
    &= 4 \pi Y_{l}^{m}(q/|q|) (-i)^{l}
    \int_{{\mathbb R}^+} r^2 R(r) \ j_{l}(|q| r) dr
   \end{aligned}
  ```
  This also holds true for real spherical harmonics.

## Spherical harmonics
- Plane wave expansion formula
  ```math
  e^{i {q} \cdot {r}} =
       4 \pi \sum_{l = 0}^\infty \sum_{m = -l}^l
       i^l j_l(|q| |r|) Y_l^m(q/|q|) Y_l^{m\ast}(r/|r|)
  ```
- Spherical harmonics orthogonality
  ```math
  \int_{\mathbb{S}^2} Y_l^{m*}(r)Y_{l'}^{m'}(r) dr
       = \delta_{l,l'} \delta_{m,m'}
  ```
  This also holds true for real spherical harmonics.

- Spherical harmonics parity
  ```math
  Y_l^m(-r) = (-1)^l Y_l^m(r)
  ```
  This also holds true for real spherical harmonics.
