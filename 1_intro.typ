#import "@preview/touying:0.6.1": *
#import themes.university: *

#import "@preview/numbly:0.1.0": numbly
#import "@preview/algo:0.3.6": algo, code, comment, d, i
#import "@preview/gentle-clues:1.2.0": *

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

== Definitions

Derivative = linear approximation of function $f$ around point $x$:

$ f(x + v) = f(x) + partial f(x) [v] + o(h) $

Here $partial f(x)[v]$ means "the linear map $partial f(x)$ applied to vector $v$".

If $f : bb(R)^n --> bb(R)^m$, then $partial f(x)$ can be represented as the Jacobian matrix:

$
  J_f (x) = ( (partial f_i) / (partial x_j) (x))_(i, j) = mat(
    (partial f_1) / (partial x_1) (x), ..., (partial f_1) / (partial x_n) (x);
    ..., ..., ...;
    (partial f_m) / (partial x_1) (x), ..., (partial f_m) / (partial x_n) (x);
  )
$

== Why differentiation?

Derivatives tell how a function input $x$ influences the output $f(x)$.

Essential for nonlinear optimization, e.g. with gradient descent.

#algo(line-numbers: false)[
  Start with initial point $x_0$ \
  For $t = 0, ..., T-1$ #i\
  $x_(t+1) = x_t - eta nabla f(x_t)$ #d\
  Return $x_T$
]

Also useful elsewhere (differential equations, sensitivity analysis).

== Why automatic?

#slide(composer: (50%, auto))[
  Deep neural networks = elementary layers + complex architecture.

  Example: #cite(<vaswaniAttentionAllYou2017>)

  Automatic differentiation enables easy experimentation & modular code.
][
  #image("img/attention.png")
]

== The big picture

From the book by #cite(<blondelElementsDifferentiableProgramming2024>):

_Differentiable programming is a programming paradigm in which complex computer programs (including those with control flows and data structures) can be differentiated end-to-end automatically, enabling gradient-based optimization of parameters in the program._

A neural network is but one kind of differentiable program.

= Flavors of differentiation

== Manual differentiation

Break down the expression for $f(x)$ into simple blocks.

$ f(x) = exp(sin(2x) + x^3) $

Human reasoning:

+ $(x^3)' = 3x^2$
+ $sin(2x)' = 2 cos(2x)$
+ $(sin(2x) + x^3)' = 2 cos(2x) + 3x^2$
+ $(exp(sin(2x) + x^3))' = (2 cos(2x) + 3x^2) exp(sin(2x) + x^3)$

Gives an expression for $f'(x)$, but takes away your will to live.

== Symbolic differentiation

#slide()[
  Plug the expression for $f(x)$ into a computer algebra system.

  Gives an expression for $f'(x)$, possibly very long.

  Expression trees are not great for computer programs:

  - intermediate variables
  - loops
][
  #image("img/expression.png")

  Graph & tree representations for $ f(x) = sin(x_1 + x_2) cos(x_1 + x_2) $
  #cite(<laueEquivalenceAutomaticSymbolic2022>)
]

== Computational graphs

Programs naturally map to directed acyclic graphs.

#image("img/computational_graph.png")

DAG for $f(x) = x_2 e^(x_1) sqrt(x_1 + x_2 e^(x_1))$ #cite(<blondelElementsDifferentiableProgramming2024>).

== Numeric differentiation

Execute the computational graph $f$ at nearby points (finite differences):

$ partial f(x)[v] approx (f(x + epsilon v) - f(x)) / epsilon $

Great at first glance:

- Applies to arbitrary programs
- Only requires two function calls instead of one

== Problems of numeric differentiation (1)

#slide(composer: (70%, auto))[
  #align(center)[
    #image("img/finite_differences.png")
  ]
][
  #align(horizon)[
    Numerical errors

    #cite(<blondelElementsDifferentiableProgramming2024>)
  ]
]

== Problems of numeric differentiation (2)

Computing a gradient is expensive: $n+1$ evaluations

$
  nabla f(x) = mat(
    partial_1 f(x); \
    partial_n f(x); \
    ...; \
    partial_n f(x)
  ) approx 1/epsilon mat(
    f(x + epsilon e_1) - f(x); \
    f(x + epsilon e_2) - f(x); \
    ...; \
    f(x + epsilon e_n) - f(x);
  )
$

== Automatic differentiation

Transform the computational graph $f$ into a new graph $partial f$.

- Keeps the compact graph encoding
- Yields exact derivative values
- Can compute gradients in just one "function call"

= Forward mode

== Derivatives as linear maps

#lorem(20)

== The chain rule

#lorem(20)

== Elementary derivatives

#lorem(20)

= Reverse mode

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

= Pitfalls

= Conclusion

== Literature pointers

- #cite(<baydinAutomaticDifferentiationMachine2018>, form: "prose")
- #cite(<margossianReviewAutomaticDifferentiation2019>, form: "prose")
- #cite(<blondelElementsDifferentiableProgramming2024>, form: "prose")

== References

#text(size: 12pt)[
  #bibliography("AD.bib", title: none, style: "apa")
]
