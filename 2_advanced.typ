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
  title: [Advanced automatic differentiation],
  subtitle: [Optimization-Augmented Machine Learning:#linebreak()Theory and Practice],
  author: [Guillaume Dalle],
  date: [2025-07-22],
  institution: [LVMT, ENPC],
))

#title-slide()

#components.adaptive-columns(outline(depth: 1))

= Motivation

== Solvers as layers

Here we love all kinds of numerical algorithms:

- stochastic simulators
- optimization solvers
- physical models
- game-theoretical equilibria

Let's use them as subroutines in a differentiable programming pipeline.

== Bilevel optimization

Bilevel optimization problems #cite(<beckGentleIncompleteIntroduction2021>) have the form

$
  min_(x in cal(X), y) thick F(x, y)
  quad "s.t." quad cases(delim: "|", thick G(x, y) >= 0, thick y in S(x))
$

where $S(x)$ is the set of optimal solutions to

$
  min_y thick f(x, y)
  quad "s.t." quad g(x, y) >= 0
$

A natural approach is to express $y(x)$ (assuming unicity) in a differentiable manner.

== Decision-focused learning

#image("img/mandi/Fig1.png", width: 100%)

See #cite(<mandiDecisionFocusedLearningFoundations2024>) (or next lecture) for a review.

== Other applications

- Reinforcement learning
- Inverse modeling
- Sensitivity analysis

== The right notion of layer

A function is approximated by a program #cite(<huckelheimTaxonomyAutomaticDifferentiation2024>). We can either

#columns[
  1. Differentiate the approximation
  #colbreak()
  2. Approximate the derivative
]

#align(center)[
  #muchpdf(read("img/huckelheim/integral.pdf", encoding: none), height: 65%)
]

== Example: square root with Heron's method

The following iteration approximates the square root of $a in [0, infinity[$:

$ x_0 = a, quad x_(n+1) = 1/2 (x_n + a/x_n) $

#columns[
  #algo(line-numbers: false, inset: 20pt)[
    $x_0 = a$ \
    $dot(x)_0 = 1$ \
    While not converged #i\ \
    $x_(n+1) = 1/2 (x_n + a/x_n)$\ \
    $dot(x)_(n+1) = 1/2 (dot(x)_n + 1/x_n - (a dot(x)_n)/(2 x_n^2))$#d\ \
    Return $dot(x)_N$
  ]

  #colbreak()

  #algo(line-numbers: false, inset: 20pt)[
    $x_0 = a$ \
    While not converged #i\ \
    $x_(n+1) = 1/2 (x_n + a/x_n)$#d\ \
    Return $1 / (2 x_N)$
  ]
]

== The chain rule is back

#columns[
  Autodiff leverages the chain rule to break down programs.

  We can circumvent it by defining custom rules for our layers.

  #colbreak()

  #muchpdf(read("img/blondel/chain_rule_recap.pdf", encoding: none))
]

Today's lecture: how to work out these rules, and what to do when they don't exist.

= Stochastic sampling layers

== Parametric expectations

Consider the parametric expectation

$
  F(theta) = bb(E)_(X tilde p(theta))[f(X)] = integral f(x) p(x, theta) "d"x
$

approximated by the Monte-Carlo estimate

$
  F_N (theta) = 1/N sum_(n=1)^N f(x^((n))) quad "where" quad x^((n)) tilde p(theta)
$

We can estimate $nabla F(theta)$ with a Monte-Carlo approach too #cite(<mohamedMonteCarloGradient2020>).

== Score function

The score function is defined by

$
  nabla_theta log p(x, theta) = (nabla_theta p(x, theta)) / p(x, theta)
$

We use it as follows:

$
  nabla F(theta) & = integral f(x) nabla_theta p(x, theta) "d"x                    \
                 & = integral [f(x) nabla_theta log p(x, theta) ] p(x, theta) "d"x \
                 & = bb(E)_(X tilde p(theta)) [f(X) nabla_theta log p(X, theta)]
$

== Score function estimator

We approximate the gradient

$
  nabla F(theta) = nabla_theta bb(E)_(X tilde p(theta))[f(X)] = bb(E)_(X tilde p(theta)) [f(X) nabla_theta log p(X, theta)]
$

with Monte-Carlo using the same distribution $p(theta)$

$
  G_n^"score" (theta) = 1/N sum_(n=1)^N f(x^((n)) nabla_theta log p(x^((n)), theta))
$

Also known as REINFORCE gradient.

== Pathwise estimator

Assume we can rewrite $X tilde p(theta)$ as $X = g(theta, Z)$ with $Z tilde q$ independent from $theta$ (decouple sampling and transformation):

$
              bb(E)_(X tilde p(theta))[f(X)] & = bb(E)_(Z tilde q) [f(g(theta, Z))]                     \
  nabla_theta bb(E)_(X tilde p(theta))[f(X)] & = bb(E)_(Z tilde q)[nabla_theta (f compose g)(theta, Z)]
$

We approximate with Monte-Carlo using a different distribution $q$:

$
  G_n^"pathwise" (theta) = 1/N sum_(n=1)^N nabla_theta (f compose g)(theta, z^((n))) quad "where" quad z^((n)) tilde q
$

Also known as the reparametrization trick.

== Reparametrizable distributions

- Gaussian: $X tilde cal(N)(mu, Sigma)$ yields $ X = mu + L Z quad "with" quad Z tilde cal(N)(0, I) quad "and" quad L L^top = Sigma $
- Exponential, Gamma: same idea
- Invert the cumulative distribution function and go back to $cal(U)(0, 1)$

== Comparison of estimators

#align(center)[
  #table(
    columns: (auto, auto, auto),
    align: horizon,
    inset: 10pt,
    table.header([], [*Score function*], [*Pathwise*]),
    [Unbiased], [yes], [yes],
    [Hypotheses on $p$], [smooth], [reparametrizable],
    [Hypotheses on $f$], [none], [smooth],
    [Variance], [high], [low],
  )
]

Score function is more widely applicable (black box $f$, discrete $p$).

Pathwise is more robust (lower variance, does not grow with dimension).

== REINFORCE variance reduction

*Rao-Blackwellisation*: condition on a subset of variables and integrate out the rest analytically.

*Control variates*: since $bb(E)_(X tilde p(theta))[nabla_theta log p(X, theta)] = 0$, we also have

$
  nabla F(theta) = bb(E)_(X tilde p(theta))[(f(X) - beta) nabla_theta log p(X, theta)]
$

for any constant baseline $beta$. More sophisticated methods allow learning the control variates (actor-critic RL).

== Discrete reparametrization

Nascent research on reparametrization for discrete distributions #cite(<aryaAutomaticDifferentiationPrograms2022>).

So far not generalized to reverse mode.

= Continuous optimization layers

== Envelope theorems

#grid(
  columns: (auto, auto),
  [
    Differentiate the value of an optimization problem #cite(<blondelElementsDifferentiableProgramming2024>):

    $
      g(theta) = max_(x in cal(X)) f(x, theta)
    $

    Danskin's or Rockafellar's theorem:

    $
      nabla g(theta) = nabla_x f(x^star (theta), theta) \ x^star (theta) = limits("argmax")_(x in cal(X)) thick f(x, theta)
    $

    under different hypotheses (convexity vs differentiability).
  ],
  [
    #muchpdf(read("img/blondel/envelope_theorem.pdf", encoding: none))
    #muchpdf(read("img/blondel/envelope_theorem_legend.pdf", encoding: none))
  ],
)

== Unrolling

Now we switch to the thornier argmax differentiation:

$
  x^star (theta) = limits("argmax")_(x in cal(X)) thick f(x, theta)
$

If we compute $x^star (theta)$ with an iterative procedure, we can unroll it: differentiate through every iteration. But...

#columns[

  - the solver may not be autodiff-friendly
  - huge memory footprint in reverse mode

  #colbreak()

  #muchpdf(read("img/blondel/chain_vjp_memory.pdf", encoding: none))
]

== Implicit function theorem

Suppose we know optimality conditions that are satisfied by $x^star (theta)$:

$
  c(x^star (theta), theta) = 0
$

Differentiating through those conditions yields:

$
  underbrace(partial_x c, A) thick underbrace(partial_theta x^star, J) + underbrace(partial_theta c, -B) = 0
$

By solving the linear system $A J = B$, we recover the Jacobian $partial_theta x^star (theta)$.

Hypotheses: smoothness, $A$ must be invertible.

== Implicit layers

Optimization (and other) layers are defined by what they return, not how they compute it.

The implicit function theorem can be inserted into autodiff #cite(<blondelEfficientModularImplicit2022>).

$ A J = B quad "with" A = partial_x c, B = -partial_theta c, J = partial_theta x^star $

#columns[
  *Forward mode*

  $A(J v) = B v$

  1. Compute $w = B v$
  2. Solve $A u = w$ for $u$

  #colbreak()

  *Reverse mode*

  $v^top J = u^top A J = u^top B$

  1. Solve $A^top u = v$ for $u$
  2. Compute $u^top B$
]

== Picking optimality conditions

Related to the algorithm used for $limits("argmax")_(x in cal(X)) thick f(x, theta)$.

#align(center)[
  #table(
    columns: (auto, auto, auto),
    align: horizon,
    inset: 15pt,
    table.header([*Constraints*], [*Algo*], [*Conditions*]),
    [$cal(X) = bb(R)^n$], [Gradient descent], [$nabla_x f(x, theta) = 0$],
    [$cal(X) = cases(G(x, theta) <= 0, H(x, theta) = 0)$], [Primal-dual], [Karush-Kuhn-Tucker],
    [$cal(X)$ projection-\ friendly],
    [Projected \ gradient descent],
    [$"proj"_(cal(X))(x - eta nabla_x f(x, theta)) = 0$],

    [$cal(X)$ LP-friendly], [Frank-Wolfe], [Projected GD on the simplex],
  )
]

More examples in #cite(<blondelEfficientModularImplicit2022>).

== Quadratic programs

#lorem(20)

== Conic programs

#lorem(20)

= Discrete optimization layers

== Branching (sigmoid)

#lorem(20)

== Choosing (softmax)

#lorem(20)

== Linear programs

#lorem(20)

== Regularization

#lorem(20)

== Perturbation

#lorem(20)

= Equilibrium layers

== Variational inequalities

#lorem(20)

== Fixed points

#lorem(20)

= Neural surrogates

== Graph neural networks

#lorem(20)

== Dynamic programming

#lorem(20)

== Large language models

#lorem(20)

= Implementations

== Python

#lorem(20)

== Julia

#lorem(20)

= What I haven't said

== Compiler tricks

The level of abstraction isn't just a mathematical question: it also matters for the compiler.

`Enzyme.jl` #cite(<mosesInsteadRewritingForeign2020>) #cite(<mosesReversemodeAutomaticDifferentiation2021>) advocates for differentiating low-level code

#columns[
  #muchpdf(read("img/moses/autodiff_pipelines.pdf", encoding: none))
  #colbreak()
  #muchpdf(read("img/moses/enzyme_approach.pdf", encoding: none))
]

== References

#text(size: 12pt)[
  #bibliography("AD.bib", title: none)
]
