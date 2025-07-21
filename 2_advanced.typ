#import "@preview/touying:0.6.1": *
#import themes.university: *

#import "@preview/numbly:0.1.0": numbly
#import "@preview/algo:0.3.6": algo, code, comment, d, i
#import "@preview/note-me:0.5.0": *
#import "@preview/mannot:0.3.0": *
#import "@preview/muchpdf:0.1.1": *
#import "@preview/fletcher:0.5.8": *

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

= Introduction

#thanks[Figures without attribution are borrowed from #cite(<blondelElementsDifferentiableProgramming2024>)]

== Solvers as layers

We love all kinds of numerical algorithms:

- stochastic simulators
- optimization solvers
- physical models
- game-theoretical equilibria

Can we use them as differentiable subroutines?

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

A natural approach is to express $y^star (x)$ (assuming unicity) in a differentiable manner.

== Decision-focused learning

#image("img/mandi/Fig1.png", width: 100%)

See #cite(<mandiDecisionFocusedLearningFoundations2024>) (or next lecture) for a review.

== Other applications

- Reinforcement learning
- Probabilistic programming
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

  Today's lecture: how to work out these rules, and what to do when they don't exist.
  #colbreak()

  #muchpdf(read("img/blondel/chain_rule_recap.pdf", encoding: none))
]

= Integration layers

== Parametric expectations

Consider the parametric expectation / integral #cite(<mohamedMonteCarloGradient2020>)

$
  f(theta) = bb(E)_(X tilde p(theta))[g(X)] = integral g(x) p(x, theta) thick "d"x
$

approximated by the Monte-Carlo estimate

$
  f_N (theta) = 1/N sum_(n=1)^N g(x^((n))) quad "where" quad x^((n)) tilde p(theta)
$

Can we find a Monte-Carlo estimate for $nabla f(theta)$ too? Not obvious:

$
  nabla f(theta) & = integral g(x) nabla_theta p(x, theta) thick "d"x
$

== Score function

The score function is defined by

$
  nabla_theta log p(x, theta) = (nabla_theta p(x, theta)) / p(x, theta)
$

We use it as follows:

$
  nabla f(theta) & = integral g(x) nabla_theta p(x, theta) thick "d"x = integral g(x) (nabla_theta p(x, theta)) / p(x, theta) p(x, theta) thick "d"x \
  & = integral [g(x) nabla_theta log p(x, theta) ] p(x, theta) thick "d"x \
  & = bb(E)_(X tilde p(theta)) [g(X) nabla_theta log p(X, theta)]
$

== Score function estimator

We approximate the gradient

$
  nabla f(theta) = nabla_theta bb(E)_(X tilde p(theta))[g(X)] = bb(E)_(X tilde p(theta)) [g(X) nabla_theta log p(X, theta)]
$

with Monte-Carlo using the same distribution $p(theta)$

$
  1/N sum_(n=1)^N g(x^((n))) nabla_theta log p(x^((n)), theta)
$

Also known as REINFORCE gradient.

== Pathwise estimator

Assume we can rewrite $X tilde p(theta)$ as $X = h(theta, Z)$ with $Z tilde q$ independent from $theta$ (decouple sampling and transformation):

$
              bb(E)_(X tilde p(theta))[g(X)] & = bb(E)_(Z tilde q) [g(h(theta, Z))]                     \
  nabla_theta bb(E)_(X tilde p(theta))[g(X)] & = bb(E)_(Z tilde q)[nabla_theta (g compose h)(theta, Z)]
$

We approximate with Monte-Carlo using a different distribution $q$:

$
  1/N sum_(n=1)^N nabla_theta (g compose h)(theta, z^((n))) quad "where" quad z^((n)) tilde q
$

Also known as the reparametrization trick.

== Reparametrizable distributions

- Gaussian: $X tilde cal(N)(mu, Sigma)$ yields
$
  X = mu + L Z quad "with" quad Z tilde cal(N)(0, I) quad "and" quad L L^top = Sigma
$
- Same idea for other location-scale distributions
- Invert the cumulative distribution function:
$
  X = F^(-1)(U) quad "where" quad F(x) = bb(P)(X <= x) quad "and" quad U tilde cal(U)(0, 1)
$

== Comparison of estimators

#align(center)[
  #table(
    columns: (auto, auto, auto),
    align: horizon,
    inset: 10pt,
    table.header([], [*Score function*], [*Pathwise*]),
    [Unbiased], [yes], [yes],
    [Consistent], [yes], [yes],
    [Hypotheses on $p$], [smooth], [reparametrizable],
    [Hypotheses on $f$], [none], [smooth],
    [Variance], [high], [low],
  )
]

Score function is more widely applicable (black box $f$, discrete $p$).

Pathwise is more robust (lower variance, stable with dimension).

== REINFORCE variance reduction

*Rao-Blackwellisation*: condition on a subset of variables and integrate out the rest analytically.

*Control variates*: since $bb(E)_(X tilde p(theta))[nabla_theta log p(X, theta)] = 0$, we also have

$
  nabla F(theta) = bb(E)_(X tilde p(theta))[(f(X) - beta) nabla_theta log p(X, theta)]
$

for any constant baseline $beta$.

More sophisticated methods allow learning the baseline (actor-critic RL).

== Discrete reparametrization

Nascent research on reparametrization for discrete distributions #cite(<aryaAutomaticDifferentiationPrograms2022>).

So far not generalized to reverse mode.

#columns[
  #muchpdf(read("img/arya/exp.pdf", encoding: none))
  #muchpdf(read("img/arya/ber1.pdf", encoding: none))
]

== Stochastic functions

What about more complicated mixes of sampling and computation?

Generalize autodiff to stochastic computation graphs #cite(<schulmanGradientEstimationUsing2015>).

#muchpdf(read("img/schulman/simple-scgs2.pdf", encoding: none))

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

  - solver may not be autodiff-friendly
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

`OptNet`, the grandfather of optimization layers #cite(<amosOptNetDifferentiableOptimization2017>)

$
  x^star underbrace((Q, q, A, b, G, h), theta) = limits("argmin")_x thick 1/2 x^top Q x + q^top x quad "s.t." quad A x = b, G x <= h
$

With dual variables $nu$ and $lambda >= 0$, the Lagrangian is:

$
  cal(L)(x, nu, lambda) = 1/2 x^top Q x + q^top x + nu^top (A x - b) + lambda^top (G x - h)
$

== Quadratic programs (2)

The KKT conditions (without inequalities) are:

$
  Q x^star + q + A^top nu^star + G^top lambda^star & = 0 \
                                      A x^star - b & = 0 \
                "diag"(lambda^star) (G x^star - h) & = 0
$

And we apply the implicit function theorem:

$
  "d" / ("d" theta) c((x^star (theta), nu^star (theta), lambda^star (theta)), theta) = 0
$

Requires the dual optimal solution as well.

== Conic programs

The `OptNet` approach generalizes to convex optimization layers reformulated as cone programs:

$
  x^star (c, A, b) = limits("argmin")_(x, s) thick c^top x quad "s.t." quad cases(delim: "|", A x + s = b, s in cal(K))
$

Key for usability: automatic reformulation with disciplined convex programming #cite(<agrawalDifferentiableConvexOptimization2019>)

== The role of strict convexity

Optimality conditions of an unconstrained optimization problem:

$
  x^star (theta) = limits("argmax")_x thick f(x, theta) quad arrow.double.long quad nabla_x f(x^star (theta), theta) = 0
$

Implicit function theorem involves the Hessian:

$
  underbrace(nabla_x^2 f, A) thick underbrace(partial_theta x^star, J) + underbrace(partial_theta nabla_x f, -B) = 0
$

Strict convexity implies existence of $(nabla_x^2 f)^(-1)$: the optimum varies smoothly with the parameter $theta$.

No longer true in the discrete case: we will need approximations!

== Is unrolling always bad?

#lorem(20)

= Discrete optimization layers

== From exact to approximate

Discrete solvers and program elements don't have useful derivatives.

#columns[

  #align(center)[
    #diagram(
      node-stroke: 1pt,
      $
        A edge(theta, ->) edge("d", 1, ->) & B edge("d", 1, ->) \
                             C edge(1, ->) & D
      $,
    )
  ]

  #colbreak()

  $
    "shortest_path"(theta) = cases(
      "ABD if" theta < 1 \
      "ACD if" theta > 1 \
      "both if" theta = 1
    )
  $

  Piecewise-constant "function"!

]

We need a nicely differentiable surrogate to allow backpropagation.

== Branching

#columns[
  #muchpdf(read("img/blondel/greater_equal_ops_hard.pdf", encoding: none))
  #muchpdf(read("img/blondel/greater_equal_ops_soft_logistic.pdf", encoding: none))

  #colbreak()

  Approximate the step function

  $
    "step"(u) = cases(1 "if" u >= 0, 0 "otherwise")
  $

  with a sigmoid function

  $
    "sigmoid"_beta (u) = 1 / (1 + e^(-beta u))
  $

  Temperature parameter $1/beta$ controls smoothness and precision.

]

== Choosing

#columns[

  Approximate $"argmax"(theta)$ with

  $
    "softmax"_beta (theta) = (exp(beta theta_i) / (sum_j exp(beta theta_j)))_i \
    "sparsemax"(theta) = limits("argmin")_(p in Delta) thick norm(p - theta)^2
  $

  Instead of picking one option, define a probability distribution.

  #colbreak()

  #muchpdf(read("img/blondel/argmax.pdf", encoding: none), height: 45%)
  #muchpdf(read("img/blondel/softargmax.pdf", encoding: none), height: 45%)

]

== Linear programs

#columns[

  #muchpdf(read("img/dalle/polytope.pdf", encoding: none))

  #colbreak()

  Focus on LPs where the cost vector $theta$ varies:

  $
    x^star (theta) = limits("argmax")_x thick theta^top x thick "s.t." thick A x <= b
  $

  Feasible set is a polyhedron.

  Almost surely on $theta$, the optimum $x^star (theta)$ is a vertex.

  Requires smoothing!
]

== Regularization

Solve a convex program instead to apply KKT implicit differentiation.

Quadratic regularization #cite(<wilderMeldingDataDecisionsPipeline2019>):

$
  x^star_gamma (theta) = limits("argmax")_x thick theta^top x - gamma norm(x)^2 thick "s.t." thick A x <= b
$

Logarithmic barrier #cite(<mandiInteriorPointSolving2020>):

$
  x^star_gamma (theta) = limits("argmax")_x thick theta^top x - gamma sum_i log(s_i) thick "s.t." thick A x + s = b
$

Must use a different solver.


== Interpolation

Replace piecewise-constant argmax with affine interpolation #cite(<vlastelicaDifferentiationBlackboxCombinatorial2020>).

$
  x^star_lambda (theta) = x^star (theta + lambda overline(x)) quad "and" quad overline(theta) = -1/lambda (x^star (theta) - x^star_lambda (theta))
$

#align(center)[
  #muchpdf(read("img/vlastelica/flambda_2D_nobox.pdf", encoding: none), width: 70%)
]

== Identity

Just pretend that the solver is the identity #cite(<sahooBackpropagationCombinatorialAlgorithms2023>)

== Perturbation

#grid(
  columns: (auto, auto),
  [
    Use REINFORCE gradient on the perturbed problem #cite(<berthetLearningDifferentiablePerturbed2020>):

    $
      x^star_epsilon (theta) = bb(E)[limits("argmax")_(x in cal(C)) thick (theta + epsilon Z)^top x]
    $

    where $Z tilde cal(N)(0, I)$ is Gaussian (other perturbation distributions are possible #cite(<dalleLearningCombinatorialOptimization2022>)).
  ],
  align(right)[
    #muchpdf(read("img/berthet/perturbed_big.pdf", encoding: none), width: 80%)
  ],
)

== Softmax tricks

#lorem(20)

== Integer linear programs

Intuitive approaches:

- Backpropagate through the continuous relaxation
- Add cutting planes #cite(<ferberMIPaaLMixedInteger2020>)

== Learning constraints

Much less explored.

- Relax them into the objective
- Approximate notion of "active constraints" for ILPs #cite(<paulusCombOptNetFitRight2021>)

= Equilibrium layers

== Nash equilibrium

#lorem(20)

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

- `cvxpylayers` #cite(<agrawalDifferentiableConvexOptimization2019>)
- `TorchOpt` #cite(<renTorchOptEfficientLibrary2023>)
- `optax` #cite(<deepmind2020jax>)
- `Theseus` #cite(<pinedaTheseusLibraryDifferentiable2022>)
- `PyEPO` #cite(<tangPyEPOPyTorchbasedEndtoend2024>)

== Julia

- `DiffOpt.jl` #cite(<sharmaFlexibleDifferentiableOptimization2022>)
- `ImplicitDifferentiation.jl` #cite(<dalleMachineLearningCombinatorial2022>)
- `DifferentiableExpectations.jl` #cite(<batyCombinatorialOptimizationDecisionfocused2025>)
- `InferOpt.jl` #cite(<dalleLearningCombinatorialOptimization2022>)

= Going further

== Inexact solves

What happens if we don't solve the problem to optimality?

#cite(<vivier-ardissonLearningLocalSearch2025>) studies the case of a local search, framed as MCMC simulation.

== Clever losses

Backpropagating through the solver itself may not be necessary.

Some loss functions provide (sometimes convex) surrogates:

- SPO+ loss #cite(<elmachtoubSmartPredictThen2022>)
- Fenchel-Young loss #cite(<blondelLearningFenchelYoungLosses2020>)
- Geometric losses #cite(<tangCaVEConeAlignedApproach2024>) #cite(<berdenSolverFreeDecisionFocusedLearning2025>)
- Learn the objective composed with the solver #cite(<zharmagambetovLandscapeSurrogateLearning2023>) #cite(<shahDecisionFocusedLearningDecisionMaking2022>)

== GPU-friendly solvers

Most (I)LP solvers run on CPU only.

Expensive back-and-forth between CPU and GPU during training.

Can we leverage parallel computing for combinatorial optimization?

- For LPs, yes #cite(<luOverviewGPUbasedFirstOrder2025>)
- For graph algorithms, maybe #cite(<yangGraphBLASTHighPerformanceLinear2022>)

Key question: whether to reuse existing algorithms or design new ones.

Inside an instance or for batching across instances?

== Compiler tricks

The level of abstraction isn't just a mathematical question: it also matters for the compiler.

`Enzyme.jl` #cite(<mosesInsteadRewritingForeign2020>) #cite(<mosesReversemodeAutomaticDifferentiation2021>) advocates for differentiating low-level code

#columns[
  #muchpdf(read("img/moses/autodiff_pipelines.pdf", encoding: none))
  #colbreak()
  #muchpdf(read("img/moses/enzyme_approach.pdf", encoding: none))
]

= Conclusion

== Take-home messages

== References

#text(size: 12pt)[
  #bibliography("AD.bib", title: none)
]
