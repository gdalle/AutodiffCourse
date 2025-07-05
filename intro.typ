#import "@preview/touying:0.6.1": *
#import themes.university: *

#import "@preview/numbly:0.1.0": numbly

#show: university-theme.with(config-info(
  title: [Introduction to automatic differentiation],
  subtitle: [Optimization-Augmented Machine Learning: Theory and Practice],
  author: [Guillaume Dalle],
  date: [2025-07-21],
  institution: [LVMT, ENPC],
))

#title-slide()

#components.adaptive-columns(outline(depth: 1))

= Motivation

= Flavors of differentiation

= Derivatives as linear maps

= Forward and reverse mode

= Jacobians and Hessians

= Implementations
