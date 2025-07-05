#import "@preview/touying:0.6.1": *
#import themes.university: *

#import "@preview/numbly:0.1.0": numbly

// #set text(font: "Fira Sans")
// #show math.equation: set text(font: "Fira Math")

#show: university-theme.with(config-info(
  title: [Introduction to automatic differentiation],
  subtitle: [Optimization-Augmented Machine Learning:#linebreak()Theory and Practice],
  author: [Guillaume Dalle],
  date: [2025-07-21],
  institution: [LVMT, ENPC],
))

#title-slide()

#components.adaptive-columns(outline(depth: 1))

= Introduction

== What is a derivative?

The best linear approximation of function $f$ around point $x$:

$ f(x + h) = f(x) + partial f(x) [h] + o(h) $

== Why do we care?

#lorem(20)

== Optimization

#lorem(20)

== Machine learning

#lorem(20)

= Flavors of differentiation

== Manual differentiation

#lorem(20)

== Symbolic differentiation

#lorem(20)

== Numeric differentiation

#lorem(20)

== Automatic differentiation

#lorem(20)

== Computational graphs

#lorem(20)

= Autodiff under the hood

== Derivatives as linear maps

#lorem(20)

== The chain rule

#lorem(20)

== Elementary derivatives

#lorem(20)

== Transposition

#lorem(20)

== Demo (JAX)

#lorem(20)

= Forward and reverse mode

== Forward mode: computing JVPs

#lorem(20)

== Reverse mode: computing VJPs

#lorem(20)

== Time complexity

#lorem(20)

== Space complexity

#lorem(20)

== Alternatives

#lorem(20)

= Jacobians and Hessians

== From linear map to matrix

#lorem(20)

== Jacobian matrix

#lorem(20)

== Hessian-Vector Product

#lorem(20)

== Hessian matrix

#lorem(20)

= Implementations

== Operator overloading

#lorem(20)

== Source transformation

#lorem(20)

== Software (Python)

#lorem(20)

== Software (Julia)

#lorem(20)

= Conclusion

== Literature pointers

- #cite(<baydinAutomaticDifferentiationMachine2018>, form: "prose")
- #cite(<margossianReviewAutomaticDifferentiation2019>, form: "prose")
- #cite(<blondelElementsDifferentiableProgramming2024>, form: "prose")

== References

#text(size: 12pt)[
  #bibliography("AD.bib", title: none, style: "apa")
]
