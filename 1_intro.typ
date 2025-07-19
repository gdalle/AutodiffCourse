#import "@preview/touying:0.6.1": *
#import themes.university: *

#import "@preview/numbly:0.1.0": numbly
#import "@preview/algo:0.3.6": algo, code, comment, d, i
#import "@preview/note-me:0.5.0": *
#import "@preview/mannot:0.3.0": *
#import "@preview/muchpdf:0.1.1": *

// #set text(font: "Fira Sans")
// #show math.equation: set text(font: "Fira Math")

#let colgray(x) = text(fill: gray, $#x$)

#let thanks(body) = {
  footnote(numbering: _ => [])[#body]
  counter(footnote).update(n => n - 1)
}

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

#thanks[Figures without attribution are borrowed from #cite(<blondelElementsDifferentiableProgramming2024>)]

== Definitions

Derivative = linear approximation of function $f$ around point $x$:

$ f(x + v) = f(x) + partial f(x) [v] + o(h) $

Here $partial f(x)[v]$ means "the linear map $partial f(x)$ applied to $v$".

#columns[
  #muchpdf(read("img/blondel/derivative.pdf", encoding: none), height: 60%)
  #colbreak()
  #v(40%)
  In the scalar case, the derivative is just a number $f'(x)$
]

== Why differentiation?

Derivatives tell how a function input $x$ influences the output $f(x)$.

Essential for nonlinear optimization, e.g. with gradient descent.

#grid(
  columns: (auto, auto),
  [
    #v(5%)
    #algo(title: "Gradient descent", line-numbers: false, parameters: ($f$, $x_0$, $T$))[
      Start with $x_0$ \
      For $t = 0, ..., T-1$ #i\
      $x_(t+1) = x_t - eta nabla f(x_t)$ #d\
      Return $x_T$
    ]
  ],
  [
    #muchpdf(read("img/blondel/gradient.pdf", encoding: none), height: 60%)
  ],
)

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
    #image("img/vaswani/ModalNet-21.png")
  ]
]

== The big picture

#columns[

  _*Differentiable programming* is a programming paradigm in which *complex computer programs* (including those with control flows and data structures) can be differentiated end-to-end automatically, enabling gradient-based *optimization of parameters* in the program._

  From the book #cite(<blondelElementsDifferentiableProgramming2024>)

  #colbreak()

  #align(center)[
    #image("img/scardapane/alice_partial.png", height: 70%)
    How deep is the \ differentiable rabbit hole #cite(<scardapaneAlicesAdventuresDifferentiable2024>)?
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

#columns[
  Plug the expression for $f(x)$ into a computer algebra system.

  Gives an expression for $f'(x)$, possibly very long.

  Expression trees are not great for computer programs:

  - intermediate variables
  - loops

  #colbreak()

  #columns[
    #muchpdf(read("img/laue/DAG.pdf", encoding: none), width: 100%)
    #muchpdf(read("img/laue/tree.pdf", encoding: none), width: 100%)
  ]

  Graph & tree representations #cite(<laueEquivalenceAutomaticSymbolic2022>)$ f(x) = sin(x_1 + x_2) cos(x_1 + x_2) $
]

== Computational graphs

Programs naturally map to directed acyclic graphs.

#muchpdf(read("img/blondel/graph_comput.pdf", encoding: none))

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
    #muchpdf(read("img/blondel/approx_error.pdf", encoding: none))
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
    partial_2 f(x); \
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

Transform the computational graph $f$ into a new graph $partial f$ and propagate derivatives through it:

- Keeps the compact program encoding ($!=$ symbolic)
- Yields exact derivative values ($!=$ numeric)
- Can compute gradients efficiently (in reverse mode)

#important[
  Automatic differentiation = chain rule + layers with known derivatives.
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

However, the linear map $v mapsto.long partial f(x)[v]$ is natural to work with:

- works in arbitrary vector spaces (and even manifolds)
- no need to materialize a matrix or flatten anything

== The chain rule

Given a function composition $f = g compose h$ with two layers, we have

$ partial f(x) = partial g(h(x)) compose partial h(x) $

The derivative of $f$ is the composition of two linear maps:

$ partial f(x): & u stretch(mapsto)^(partial h(x)) v stretch(mapsto)^(partial g( h(x) )) w $

We can differentiate any function knowing the derivatives of its layers.

== Scalar layers

Let $dot(x)$ denote an arbitrary input tangent (directional derivative).

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

For the inverse, a Jacobian matrix of size $(n times n)^2$ is not needed: the linear map is more efficient.

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

- cannot recurse into (because it is implemented in another language).
- should not recurse into (because we have a better formula).

The layer boundary is a subjective choice (see lecture 2).

== This was forward mode

Propagate the input and its tangent together through a chain of layers.

#align(center)[
  #muchpdf(read("img/blondel/chain_jvp.pdf", encoding: none), height: 60%)
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

The adjoint of a linear map $ell: bb(R)^n -> bb(R)^m$ is the only linear map $ell^* : bb(R)^m -> bb(R)^n$ such that

$ forall (x, y) in bb(R)^n times bb(R)^m, quad angle.l ell(x), y angle.r = angle.l x, ell^*(y) angle.r $

If $ell$ is represented by a matrix $A$, then $ell^*$ is represented by $A^T$.

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

We can differentiate any function in reverse mode knowing the adjoint derivatives of its layers.

== Chain rule recap

#muchpdf(read("img/blondel/chain_rule_recap.pdf", encoding: none), width: 100%)

== Back to our layer examples

Let $overline(y)$ be an arbitrary output cotangent (sensitivity).

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    align: horizon,
    inset: 10pt,
    table.header([*variables*], [*output*], [*function* $f(x)$], [*adjoint* $partial f(x)^*[overline(y)]$]),
    [$x in bb(R)$], [$y in bb(R)$], [$sin(x)$], [$cos(x)overline(y)$],
    [$x in bb(R)^n$], [$y in bb(R)^m$], [$A x$], [$A^T overline(y)$],
    [$X in bb(R)^(n times n)$], [$Y in bb(R)^(n times n)$], [$X^(-1)$], [$-X^(-T) overline(Y) X^(-T)$],
    [$X in bb(R)^(n times n)$], [$y in bb(R)$], [$log det(X)$], [$overline(y) X^(-T)$],
  )
]

More crazy formulas in #cite(<gilesExtendedCollectionMatrix2008>).

== This was reverse mode

Propagate the input through a chain of layers, record enough information, backpropagate the output cotangent.

#align(center)[
  #muchpdf(read("img/blondel/chain_vjp.pdf", encoding: none), height: 70%)
]

= Forward versus reverse

== Time complexity

*Theorem (Baur-Strassen):*
- Cost of 1 JVP (forward mode) $prop$ cost of 1 function call
- Cost of 1 VJP (reverse mode) $prop$ cost of 1 function call

Rather easy to believe:

- Individual derivatives not much harder than the corresponding layer
- Composition of linear maps adds up their computational costs

== Time complexity (special cases)

Assume the function $f$ can be computed in time $O(tau)$.

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    align: horizon,
    inset: 10pt,
    table.header([*setting*], [*object*], [*forward mode*], [*reverse mode*]),
    $bb(R)^n -> bb(R)^m$, [Jacobian matrix], $O(n tau)$, $O(m tau)$,
    $bb(R)^n -> bb(R)$, [gradient vector], $O(n tau)$, $O(tau)$,
    $bb(R) -> bb(R)^m$, [derivative vector], $O(tau)$, $O(m tau)$,
  )
]

== Space complexity

#columns[
  #align(center)[
    #muchpdf(read("img/blondel/chain_jvp_memory.pdf", encoding: none))

    Forward mode has constant memory cost.
  ]

  #colbreak()

  #align(center)[
    #muchpdf(read("img/blondel/chain_vjp_memory.pdf", encoding: none))

    Reverse mode has linear memory cost (in the depth of the chain).
  ]
]

== Saving memory

Strategies like checkpointing or reversibility can save memory at the cost of additional compute.

#align(center)[
  #muchpdf(read("img/blondel/binomial_checkpointing.pdf", encoding: none), height: 50%)
]

Alternatives to reverse mode have been suggested based on randomized forward mode #cite(<baydinGradientsBackpropagation2022>).

= Higher order

== Hessian matrices

The Hessian matrix is very useful for second-order optimization.

#grid(
  columns: (auto, auto),
  [
    #v(5%)
    #algo(title: "Newton's method", line-numbers: false, parameters: ($f$, $x_0$, $T$))[
      Start with $x_0$ \
      For $t = 0, ..., T-1$ #i\
      $x_(t+1) = x_t - eta nabla^2 f(x_t)^(-1) nabla f(x_t)$ #d\
      Return $x_T$
    ]
  ],
  [
    #muchpdf(read("img/blondel/second_der.pdf", encoding: none), height: 60%)
  ],
)

== Hessian-Vector Products

Autodiff gives us first-order derivatives: how to compose them?

The Hessian is the Jacobian of the gradient: if $f: bb(R)^n -> bb(R)$ then

$ nabla^2 f(x) = J_(nabla f)(x) $

An HVP can be computed as a JVP of the gradient (itself a VJP)

$ nabla^2 f(x)[v] = partial (nabla f)(x)[v] $

This is called forward-over-reverse mode #cite(<pearlmutterFastExactMultiplication1994>).

== Sparsity

#columns[
  In high dimensions, can't afford to compute or store the entire Hessian.

  If there are few nonzero coefficients, its sparsity can be leveraged to perform fewer HVPs.

  #colbreak()

  #align(center)[
    #muchpdf(read("img/montoison/graph_sym.pdf", encoding: none))

    Link between sparse autodiff and graph coloring #cite(<gebremedhinWhatColorYour2005>)
  ]
]

== Iterative solvers

To inverse the Hessian in Newton's method, one can use iterative linear solvers instead of factorization-based solvers #cite(<dagreouHowComputeHessianvector2024>).

#columns[
  #align(center)[
    *Pros*

    Only requires lazy HVPs

    Very GPU-friendly
  ]
  #colbreak()
  #align(center)[
    *Cons*

    Fewer iterations $->$ lower precision
  ]
]

= Implementation

== Non-standard interpretation

Autodiff reinterprets a computer program to mean something else #cite(<margossianReviewAutomaticDifferentiation2019>).

We can take the same idea in other directions:

- Uncertainty propagation
- Physical unit checking
- Sparsity detection

== Operator overloading

Pass augmented values to language operators & overload their behavior.

Example: `Dual` numbers in `ForwardDiff.jl` #cite(<revelsForwardModeAutomaticDifferentiation2016>).

#text(
  size: 1em,
)[
  #columns[
    ```julia
    using ForwardDiff: Dual

    u, u̇ = 2.0, 3.0
    v, v̇ = 4.0, 5.0
    du = Dual(u, u̇)
    dv = Dual(v, v̇)
    ```
    #colbreak()
    ```julia
    julia> du * dv
    Dual{Nothing}(8.0,22.0)

    julia> u * v̇ + v * u̇
    22.0

    julia> du / dv
    Dual{Nothing}(0.5,0.125)

    julia> (u̇ * v - v̇ * u) / v^2
    0.125
    ```
  ]
]

== Source transformation

Preprocess the source code to add derivative bookkeeping.

Example: #raw("jaxpr") intermediate representation in #raw("JAX") #cite(<bradburyJAXComposableTransformations2018>)

#text(
  size: 0.6em,
)[
  #columns[
    ```python
    import jax
    import jax.numpy as jnp

    def selu(x, alpha=1.67, lambda_=1.05):
        return lambda_ * jnp.where(
            x > 0,
            x,
            alpha * jnp.exp(x) - alpha
        )

    x = jnp.arange(5.0)
    jax.make_jaxpr(selu)(x)
    ```
    #colbreak()
    ```python
    { lambda ; a:f32[5]. let
        b:bool[5] = gt a 0.0:f32[]
        c:f32[5] = exp a
        d:f32[5] = mul 1.6699999570846558:f32[] c
        e:f32[5] = sub d 1.6699999570846558:f32[]
        f:f32[5] = jit[
          name=_where
          jaxpr={ lambda ; b:bool[5] a:f32[5] e:f32[5]. let
              f:f32[5] = select_n b e a
            in (f,) }
        ] b a e
        g:f32[5] = mul 1.0499999523162842:f32[] f
      in (g,) }
    ```
  ]
]

== Software

#columns[
  *Python libraries*

  - #link("https://github.com/HIPS/autograd")[#raw("autograd")]
  - #link("https://www.tensorflow.org/")[#raw("TensorFlow")]
  - #link("https://pytorch.org/")[#raw("PyTorch")]
  - #link("https://docs.jax.dev/en/latest/")[#raw("JAX")]

  #colbreak()

  *Julia libraries* (see #cite(<dalleCommonInterfaceAutomatic2025>))

  - #link("https://github.com/JuliaDiff/ForwardDiff.jl")[#raw("ForwardDiff.jl")]
  - #link("https://github.com/FluxML/Zygote.jl")[#raw("Zygote.jl")]
  - #link("https://github.com/EnzymeAD/Enzyme.jl")[#raw("Enzyme.jl")]
  - #link("https://github.com/chalk-lab/Mooncake.jl")[#raw("Mooncake.jl")]
]

#align(center)[
  #columns[
    #muchpdf(read("img/dalle/ecosystem_python.pdf", encoding: none))
    #muchpdf(read("img/dalle/ecosystem_julia_di.pdf", encoding: none))
  ]
]

= What I haven't said

== Complex numbers

The derivative of a function $f : bb(C) -> bb(C)$ is a matrix $f'(x) in bb(R)^(2 times 2)$.

For holomorphic functions, we can identify it with a complex number, otherwise not.

Different autodiff frameworks have different notions of complex gradients #cite(<kramerTutorialAutomaticDifferentiation2024>)!

== Branches and loops

Autodiff returns a locally valid derivative, for the path taken by the primal program.

- #raw("if") / #raw("else"): specific to the chosen branch
- #raw("for") / #raw("while"): specific to the number of iterations

Some autodiff frameworks error on code with value-dependent control flow.

== Mutation

For efficiency, many numerical programs reuse memory.

Some autodiff frameworks error on code with mutation.

== Non-smoothness

Autodiff acts on programs and not functions #cite(<bolteMathematicalModelAutomatic2020>).

The following programs give different derivatives at 0:

$"relu"(t) = max(0, t) quad "relu"_2(t) = "relu"(-t) + t quad "relu"_3(t) = ("relu"(t) + "relu"_2(t)) / 2$

#align(center)[
  #muchpdf(read("img/bolte/3reluActivations.pdf", encoding: none), width: 100%)
]

== Approximations

Computer programs are approximations of mathematical functions #cite(<huckelheimTaxonomyAutomaticDifferentiation2024>).

For example, $f(A, b) = A^(-1) b$ is obtained via a program $p(A, b) approx A^(-1) b$.

What should the computed "automatic" derivative be?

1. Differentiate the approximation $partial p(X)$
2. Approximate the derivative $hat(partial f(X))$

More on this next time!

== References

#text(size: 12pt)[
  #bibliography("AD.bib", title: none)
]
