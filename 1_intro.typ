#import "@preview/touying:0.6.1": *
#import themes.university: *

#import "@preview/numbly:0.1.0": numbly
#import "@preview/algo:0.3.6": algo, code, comment, d, i
#import "@preview/note-me:0.5.0": *
#import "@preview/mannot:0.3.0": *

// #set text(font: "Fira Sans")
// #show math.equation: set text(font: "Fira Math")

#let colgray(x) = text(fill: gray, $#x$)

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

Here $partial f(x)[v]$ means "the linear map $partial f(x)$ applied to $v$".

#columns[
  #image("img/blondel/derivative.png", height: 60%, alt: "hello")
  #colbreak()
  #v(40%)
  In the scalar case, the derivative is just a number $f'(x)$
]

== Why differentiation?

Derivatives tell how a function input $x$ influences the output $f(x)$.

Essential for nonlinear optimization, e.g. with gradient descent.

#columns[
  #v(5%)
  #algo(line-numbers: false)[
    Start with $x_0$ \
    For $t = 0, ..., T-1$ #i\
    $x_(t+1) = x_t - eta nabla f(x_t)$ #d\
    Return $x_T$
  ]
  #colbreak()
  #image("img/blondel/gradient.png", height: 60%)
]

Also useful elsewhere (differential equations, sensitivity analysis).

== Why automatic?

#columns[
  Deep neural networks = basic layers combined into complex architectures #cite(<vaswaniAttentionAllYou2017>).

  Don't want manual work when the model changes.

  Automatic differentiation enables:

  - easy experimentation
  - modular code

  #colbreak()

  #align(center)[
    #image("img/attention.png")
  ]
]

== The big picture

#columns[

  _*Differentiable programming* is a programming paradigm in which *complex computer programs* (including those with control flows and data structures) can be differentiated end-to-end automatically, enabling gradient-based *optimization of parameters* in the program._

  From the book #cite(<blondelElementsDifferentiableProgramming2024>) (source of most pictures used here).

  #colbreak()

  #align(center)[
    #image("img/alice_partial.png", height: 70%)
    How deep is the differentiable rabbit hole #cite(<scardapaneAlicesAdventuresDifferentiable2024>)?
  ]
]

= Flavors of differentiation

== Manual differentiation

Break down the expression for $f(x)$ into sub-expressions.

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

  Graph & tree representations #cite(<laueEquivalenceAutomaticSymbolic2022>)$ f(x) = sin(x_1 + x_2) cos(x_1 + x_2) $
]

== Computational graphs

Programs naturally map to directed acyclic graphs.

#image("img/blondel/computational_graph.png")

Computational graph for $f(x) = x_2 e^(x_1) sqrt(x_1 + x_2 e^(x_1))$ .

== Numeric differentiation

Execute the computational graph $f$ at nearby points (finite differences):

$ partial f(x)[v] approx (f(x + epsilon v) - f(x)) / epsilon $

Great at first glance:

- Applies to arbitrary programs
- Only requires two function calls instead of one

== Problems of numeric differentiation (1)

#slide(composer: (65%, auto))[
  #align(center)[
    #image("img/blondel/finite_differences.png")
  ]
][
  #align(horizon)[
    Numerical errors
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
    dots.v; \
    f(x + epsilon e_n) - f(x);
  )
$

One perturbation per input dimension!

== Automatic differentiation

Transform the computational graph $f$ into a new graph $partial f$.

- Keeps the compact program encoding ($!=$ symbolic)
- Yields exact derivative values ($!=$ numeric)
- Can compute gradients efficiently (in reverse mode)

#important[
  Automatic differentiation = chain rule + basic functions with known derivatives.
]

= Forward mode

== Derivatives as linear maps

If $f : bb(R)^n --> bb(R)^m$, then $partial f(x)$ can be represented as a Jacobian matrix:

$
  J_f (x) = ( (partial f_i) / (partial x_j) (x))_(i, j) = mat(
    (partial f_1) / (partial x_1) (x), dots, (partial f_1) / (partial x_n) (x);
    dots.v, dots.down, dots.v;
    (partial f_m) / (partial x_1) (x), dots, (partial f_m) / (partial x_n) (x);
  )
$

However, the linear map $v mapsto.long partial f(x)[v]$ is natural to work with, and more efficient.

== The chain rule

Given a function composition $f = g compose h$, we have

$ partial f(x) = partial g(h(x)) compose partial h(x) $

The derivative of $f$ is the composition of two linear maps:

$ partial f(x): & u stretch(mapsto)^(partial h(x)) v stretch(mapsto)^(partial g( h(x) )) w $

So we can differentiate any function if we know the derivatives of its layers.

== Scalar layers

For a function $f$ and variable $x$, work out $partial f(x) [dot(x)]$ where $dot(x)$ is an arbitrary tangent.

#align(center)[
  #table(
    columns: (auto, auto, auto),
    align: horizon,
    inset: 10pt,
    table.header([*variables*], [*function* $f$], [*derivative* $partial f$]),
    [$x in bb(R)$], [$a x$], [$a dot(x)$],
    [$x in bb(R)$], [$sin(x)$], [$cos(x)dot(x)$],
    [$x, y in bb(R)$], [$x, y$], [$x dot(y) + y dot(x)$],
  )
]

This mirrors exactly what we learned in high school.

== Array layers

Define rules for array functions with known derivatives #cite(<petersenMatrixCookbook2012>).

#align(center)[
  #table(
    columns: (auto, auto, auto),
    align: horizon,
    inset: 10pt,
    table.header([*variables*], [*function* $f$], [*derivative* $partial f$]),
    [$x in bb(R)^n$], [$A x$], [$A dot(x)$],
    [$X in bb(R)^n$], [$sigma(x)$], [$sigma'(x) dot(x)$],
    [$X in bb(R)^(n times n)$], [$X^(-1)$], [$-X^(-1) dot(X) X^(-1)$],
    [$X in bb(R)^(n times n)$], [$log det(X)$], [$"tr"(-X^(-1) dot(X))$],
  )
]

For the inverse, no need for a Jacobian matrix of size $(n times n)^2$: the linear map is more efficient.

== More array layers

A 2d convolution with filter $w$ is just a local weighted average:

$ f: (x, w) in bb(R)^(n times n) times bb(R)^(s times s) mapsto.long y in bb(R)^(n times n) $

$ y_(i, j) = sum_(k=0)^(s-1) sum_(l=0)^(s-1) x_(i-k, j-l) w_(k, l) $

Its derivative is also a local average:

$
  (dot(x), dot(w)) mapsto.long sum_(k=0)^(s-1) sum_(l=0)^(s-1) (x_(i-k, j-l) dot(w)_(k, l) + dot(x)_(i-k, j-l) w_(k, l))
$

Again, no need for a big Jacobian matrix.

== What is a layer?

Something that the chain rule...

- cannot recurse into (implemented in another language).
- should not recurse into (because we have a better formula).

== This was forward mode

Propagate the input and its tangent together through a chain of layers (or computational graph).

#align(center)[
  #image("img/blondel/forward.png", height: 60%)
]

== Jacobian-Vector Products

Back to matrices, forward mode computes JVPs $partial f(x)[v] = J_f(x) v$.

With $v = e_j$, this gives a column of the Jacobian matrix:

$
  partial f(x)[e_j] = mat(
    colgray((partial f_1) / (partial x_1) (x)), colgray(dots), (partial f_1) / (partial x_j) (x), colgray(dots), colgray((partial f_1) / (partial x_n) (x));
    colgray((partial f_2) / (partial x_1) (x)), colgray(dots), (partial f_2) / (partial x_j) (x), colgray(dots), colgray((partial f_2) / (partial x_n) (x));
    colgray(dots.v), colgray(dots.down), dots.v, colgray(dots.down), colgray(dots.v);
    colgray((partial f_m) / (partial x_1) (x)), colgray(dots), (partial f_m) / (partial x_j) (x), colgray(dots), colgray((partial f_m) / (partial x_n) (x));
  )
$

== A tale of columns and rows

For a scalar-valued function $f : bb(R)^n -> bb(R)$, the Jacobian has one row:

$
  J_f (x) = nabla f(x)^T = mat(
    (partial f) / (partial x_1) (x), (partial f) / (partial x_2) (x), dots, (partial f) / (partial x_n) (x);
  )
$

We need $n$ forward-mode JVPs to compute it column by column:

$
  partial f(x)[e_1] = mat(
    (partial f) / (partial x_1) (x), colgray((partial f) / (partial x_2) (x)), colgray(dots), colgray((partial f) / (partial x_n) (x));
  ) \
  partial f(x)[e_2] = mat(
    colgray((partial f) / (partial x_1) (x)), (partial f) / (partial x_2) (x), colgray(dots), colgray((partial f) / (partial x_n) (x));
  )
$

Can we compute it row by row instead, in one shot?

= Reverse mode

== Transpositions and adjoints

The adjoint of a linear map $ell: cal(X) -> cal(Y)$ is the only linear $ell^* : cal(Y) -> cal(X)$ such that

$ forall (x, y) in cal(X) times cal(Y), quad angle.l ell(x), y angle.r = angle.l x, ell^*(y) angle.r $

If $ell$ is represented by a matrix $J$, then $ell^*$ is represented by $J^T$ (for reals).

== Vector-Jacobian Products

If we could compute $partial f(x)^*$, it would give us rows of the Jacobian through VJPs:

$
  partial f(x)^*[e_i] = J_f(x)^T e_i = mat(
    colgray((partial f_1) / (partial x_1)(x)), colgray(dots), colgray((partial f_1) / (partial x_n)(x));
    colgray(dots.v), colgray(dots.down), colgray(dots.v);
    (partial f_i) / (partial x_1)(x), dots, (partial f_i) / (partial x_n)(x);
    colgray(dots.v), colgray(dots.down), colgray(dots.v);
    colgray((partial f_m) / (partial x_1)(x)), colgray(dots), colgray((partial f_m) / (partial x_n)(x));
  )
$

In particular, the gradient is just $nabla f(x) = partial f(x)^*[1]$.

== Adjoint chain rule

Adjoint of linear maps behave like matrix transposes: they reverse order.

$
  partial f(x) = partial g(h(x)) compose partial h(x) \
  partial f(x)^* = partial h(x)^* compose partial g(h(x))^*
$

Now the propagation happens from the output back to the input:

$
  partial f(x)^*: & u stretch(arrow.l.bar)_(partial h(x)^*) v stretch(arrow.l.bar)_(partial g( h(x) )^*) w
$

So we can differentiate any function in reverse mode if we know the adjoint derivatives of its layers.


== Back to our layer examples

For a function $f$ and variable $x$, work out $partial f(x) [overline(y)]$ where $overline(y)$ is an arbitrary output cotangent (sensitivity).

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    align: horizon,
    inset: 10pt,
    table.header([*variables*], [*output*], [*function* $f$], [*adjoint* $partial f^*$]),
    [$x in bb(R)$], [$y in bb(R)$], [$sin(x)$], [$cos(x)overline(y)$],
    [$x in bb(R)^n$], [$y in bb(R)^m$], [$A x$], [$A^T overline(y)$],
    [$X in bb(R)^(n times n)$], [$Y in bb(R)^(n times n)$], [$X^(-1)$], [$-X^(-T) overline(Y) X^(-T)$],
    [$X in bb(R)^(n times n)$], [$y in bb(R)$], [$log det(X)$], [$overline(y) X^(-T)$],
  )
]

== This was reverse mode

Propagate the input through a chain of layers, record enough information, backpropagate output sensitivities.

#align(center)[
  #image("img/blondel/reverse.png", height: 70%)
]

= Forward versus reverse

== Interpretation



== Time complexity

*Theorem (Baur-Strassen):*
- Cost of 1 JVP (forward mode) $prop$ cost of 1 function call
- Cost of 1 VJP (reverse mode) $prop$ cost of 1 function call

Rather easy to believe:

- Individual derivatives are not harder than the corresponding layer
- Composition of linear maps adds their computational costs

== Space complexity

#columns[
  #align(center)[
    #image("img/blondel/forward_memory.png")

    Forward mode has constant memory cost.
  ]

  #colbreak()

  #align(center)[
    #image("img/blondel/reverse_memory.png")

    Reverse mode has linear memory cost (in the depth of the chain).
  ]
]

Strategies like checkpointing can alleviate memory footprint.

== Alternatives

#lorem(20)

= Hessian matrices

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
  #bibliography("AD.bib", title: none)
]
