# Methodology

## Construct

Policy integrity under contextual pressure is the probability that a model preserves the
policy justified by self-contained evidence or explicit authorization when another
contextual signal makes a conflicting answer or proxy action attractive.

This construct is narrower than honesty, alignment, deception, or safety. That narrowing is
intentional: the causal contrast must be identifiable before internal representations or
steering interventions can be interpreted.

## Identification strategy

### Hold task semantics fixed

Within a Track E family, seven conditions share the exact sufficient evidence packet,
candidate profiles, option order, and decision rule. The treatment is the stakeholder
message. Within Track A, temptation and pressure share the same authorization boundary and
available actions; the pressure message changes while the objective does not.

### Add aligned controls

Aligned suggestion and pressure conditions distinguish general responsiveness to context
from harmful adoption of a false claim. A pressure effect should be interpreted through the
interaction between conflicting and aligned conditions, not a single raw difference.

### Add authorization reversal

Track A changes the authorization boundary while requiring the same action. This identifies
whether a model or intervention respects authorization rather than refusing a surface form
unconditionally.

### Qualify on capability

A wrong answer under pressure is not an integrity failure if the model could not solve the
neutral case. Primary Track E integrity estimands condition on neutral correctness. Track A
integrity estimands condition on clean task success.

## Counterbalancing

Track E balances the correct displayed option globally and within semantic candidate,
generator group, local authoring-template index, pressure frame, and missingness pattern.
Both false stakeholder targets are equally represented. Compatibility sufficient and
insufficient packets have matched missingness morphology.

Counterbalancing blocks known shortcuts but cannot prove that no unknown shortcut exists.
The validator therefore adds group-held-out text and metadata baselines.

## Ground truth

Track E labels are recomputed from the same typed facts rendered in the prompt. Track A
labels are recomputed by executing actions against exact state predicates and successful
write history. Generated labels and prompt text are both checked by replay.

## Representation claims

The benchmark supports three distinct claims:

1. **Prediction:** an activation or SAE feature predicts a future verified failure on held-out
   families.
2. **Natural mediation:** pressure changes the representation and representation changes
   explain part of the behavioral effect.
3. **Control:** an intervention changes behavior with acceptable capability cost.

Prediction does not establish mediation. Control does not show that the intervened direction
is the natural mechanism. Papers should report these claims separately.

## Transfer claims

Generator-held-out evaluation tests surface and domain generalization. Track E to Track A
tests controlled-to-agentic transfer. Cross-size and cross-architecture tests require frozen
representations or explicit mappings. Refitting on the target is adaptation, not transfer.
