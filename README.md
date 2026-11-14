# faa-attribution-forecast

Forecast Polymarket's FAA outage attribution prices, minute by minute, 4 hours
ahead — an entry for make.fun's `polymarket-faa-attribution-02` bounty.

`main.ts` (Node 22, standard library only, no network) reads one case file at
a time from the input directory and writes, for every market of the case, five
quantiles (5%, 25%, 50%, 75%, 95%) of its Yes price at each of the 240 minutes
after the checkpoint.

The forecast is a two-scale random walk around the last price: the median
stays at the checkpoint price (prediction-market prices are near-martingales,
so no drift), while the spread widens with the square root of the horizon —
tight for the inner quantiles, wide for the outer ones, because these prices
sit still most minutes and then jump. Both scales are estimated from the
case's own last 24 hours, pooled across its markets (one family's markets
share one news flow) as quantiles of the level-scaled one-minute moves
`|dp| / sqrt(p (1-p))`, with floors and caps so a quiet window never promises
a narrower band than the jumps this family has already shown.

## Run it locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

On the practice cases this scores a mean pinball loss of 0.305723, against
the house baseline's 0.363589 (median error 0.767503, 90% coverage 0.964).
