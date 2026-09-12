# What a Graph Consumed, and What It Cost

FlakeGraph records every provider call a run makes — prompt and completion
tokens, pages parsed, which model did the work, and whether it ran on hosted or
local infrastructure — and turns that record into cost using a rate card supplied
in configuration.

The separation matters. **Usage is measured; cost is derived.** A recorded price
becomes wrong the moment a provider changes its rates, and it becomes wrong
silently. Recorded usage stays true, so correcting the rate table reprices every
historical graph without re-running anything.

## What Is Recorded

One event per provider call, not one per stage. A stage that fans out across
document windows reports what each window actually consumed, so a run that
retried a window five times shows five calls rather than one.

| Field | Why it is there |
| --- | --- |
| `stage`, `operation` | Attributes spend to the work that caused it |
| `provider`, `model` | Selects the rate, and survives a later model change |
| `usage` | Prompt, completion and total tokens |
| `pages` | Document parsing bills per page, not per token |
| `locality` | Hosted work is billed; local work is *avoided* |
| `calls` | One event may cover a batched call |

Cache hits are deliberately **not** recorded. A cache hit returns the same
artifact without calling anybody, so counting it would make re-running an
unchanged corpus look as expensive as the first run — destroying the comparison
the cache exists to enable.

A call that fails after consuming tokens *is* recorded. It was billed.

## Rates

Rates live in the `consumption` block of a FlakeGraph config:

```yaml
consumption:
  usd_per_credit: 3.5
  local_reference: snowflake_cortex:mistral-large2
  rates:
    snowflake_cortex:mistral-large2:
      prompt_credits_per_million: 1.20
      completion_credits_per_million: 3.60
    snowflake_cortex:snowflake_cortex-ocr:
      credits_per_page: 0.00068
```

Two units, because providers bill in two. An hourly API quotes dollars;
Snowflake quotes **AI credits**, and the dollar value of a credit is a property
of the account's contract rather than of the model. Credits are converted through
the single `usd_per_credit` figure, so correcting one number reprices every
Snowflake model at once instead of leaving thirty hand-multiplied figures to
drift apart.

`local_reference` names the hosted model a local run is priced against. A local
run has no invoice, so its saving is only meaningful relative to a stated
alternative — without one, "avoided" would be an unfalsifiable number, and the
totals report zero rather than guess.

### Reading Snowflake's Table Correctly

The shipped card in `configs/app-defaults.yaml` covers every model Snowflake
currently serves, taken from its Service Consumption Table. Two traps are worth
knowing, because both produce plausible numbers that are wrong:

**Table 6(a) prices the SQL functions; Table 6(c) prices the REST API.** Several
models appear in both at different rates. FlakeGraph's adapters issue
`AI_COMPLETE`, `AI_EMBED` and `AI_PARSE_DOCUMENT` as SQL, so 6(a) applies. Using
6(c) for `llama3.1-70b` under-reports it by about 20%.

**`AI_PARSE_DOCUMENT` bills its two modes very differently** — Layout costs more
than five times OCR. The pipeline records which mode ran, as
`snowflake_cortex-layout` or `snowflake_cortex-ocr`, so the card can price them
apart. A single entry has to pick one and be wrong about the other.

Every model is listed, not only the ones a given deployment uses, because the
model is free text in the app. An unlisted model is reported as an **unpriced
call** rather than as spend.

## Unpriced Calls Are Reported, Not Hidden

`ConsumptionTotals.unpriced_calls` counts calls the card could not price, and a
rate only counts as covering an event if it prices the dimension the event
actually used — a card with token rates but no page rate does not silently value
a thousand parsed pages at nothing.

A total that quietly omits what it could not price reads as authoritative while
being wrong. The gap is made visible and the operator decides whether it matters.

## Reading It During A Run

The app's Consumption tab reports totals while a run is still going, broken down
by stage, provider, model and locality. Waiting until a run finishes to find out
what it cost is the wrong time to learn it.

The full report persists beside the finished graph, carrying every recorded event
rather than only the totals. That is what makes the answer re-derivable when the
rate table is corrected, and what lets spend be broken down afterwards along an
axis nobody thought to group by at the time.
