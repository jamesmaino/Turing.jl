module Variational

import AdvancedVI
import Bijectors
import DistributionsAD
import DynamicPPL
import StatsBase
import StatsFuns

import Random

# Reexports
using AdvancedVI: vi, ADVI, ELBO, elbo, TruncatedADAGrad, DecayedADAGrad
export
    vi,
    ADVI,
    ELBO,
    elbo,
    TruncatedADAGrad,
    DecayedADAGrad

"""
    make_logjoint(model::Model; weight = 1.0)

Constructs the logjoint as a function of latent variables, i.e. the map z → p(x ∣ z) p(z).
The weight used to scale the likelihood, e.g. when doing stochastic gradient descent one needs to
use `DynamicPPL.MiniBatch` context to run the `Model` with a weight `num_total_obs / batch_size`.

# Notes
- For sake of efficiency, the returned function is closes over an instance of `VarInfo`. This means that you *might* run into some weird behaviour if you call this method sequentially using different types; if that's the case, just generate a new one for each type using `make_logjoint`.
"""
function make_logjoint(model::DynamicPPL.Model, weight = 1.0)
    # setup
    ctx = DynamicPPL.MiniBatchContext(
        DynamicPPL.DefaultContext(),
        weight
    )
    varinfo_init = DynamicPPL.VarInfo(model, ctx)

    f = DynamicPPL.LogDensityFunction(varinfo_init, model, ctx)
    # TODO: Make use of `ADgradient` here instead.
    return Base.Fix1(LogDensityProblems.logdensity, f)
end

# objectives
function (elbo::ELBO)(
    rng::Random.AbstractRNG,
    alg::AdvancedVI.VariationalInference,
    q,
    model::DynamicPPL.Model,
    num_samples;
    weight = 1.0,
    kwargs...
)
    return elbo(rng, alg, q, make_logjoint(model, weight), num_samples; kwargs...)
end

# VI algorithms
include("advi.jl")

end
