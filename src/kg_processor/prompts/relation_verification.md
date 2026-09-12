Verify whether each graph triple is entailed by its quoted source evidence.
Judge the exact directed predicate, not merely whether both entities appear.
Also verify that source_surface and target_surface denote their assigned entity IDs.
For an is_document_context entity, a supplied paper surface such as 'we' or
'this paper' denotes the source document only when the quote uses it as the
grammatical actor of the asserted predicate. A supplied technical surface such
as 'our models', 'the method', or result-table 'Ours' may denote a focal METHOD
or MODEL only when exactly one ontology-compatible focal technical entity can
fill that grammatical role. Reject the mapping when several focal entities are
compatible.
The source PAPER marked is_document_context may additionally be implicit through
document provenance when an authorial technical passage states that paper's own
method, experiment, result, or contribution without repeating the title or a
first-person phrase. This exception is source-only and PAPER-only. Treat a cited,
historical, or generic background statement as insufficient rather than assigning
it to the source paper.
Supported means the quote directly justifies source, relation, and target.
When the configured lineage predicate is BUILDS_ON, wording that identifies the
source as a variant of the target or says it inherits or combines the target's
ideas, advantages, properties, or capabilities directly supports that lineage.
Do not require the literal words "builds on" when the same relation is explicit.
Distinguish a paper's discourse voice from a uniquely anteceded model or method
that actually bears a training, evaluation, implementation, or design claim.
Generic or collective surfaces such as 'the model', 'both models', or 'our
approach' are valid only when an explicit antecedent in the same one-to-three-
sentence quote uniquely identifies the assigned endpoint.
Likewise, 'the same dataset' may denote a target only when immediately preceding
bounded context names exactly one compatible dataset and the current statement
explicitly applies it to the assigned source.
When an optimizer or training technique is applied to fit a model, the model
USES_METHOD the technique even when the sentence grammatically says the authors
evaluate the technique "on" that model. For "compare X to train Y", support
Y USES_METHOD X and reject the inverse X USES_METHOD Y unless separate wording
says X operationally incorporates Y. First-person experimental framing can
separately support PAPER USES_METHOD model when the paper's authors explicitly
investigate, train, or learn that named model.
Result-table and figure-caption labels directly support method-or-model to dataset
evaluation when they unambiguously identify the measured series and its dataset.
The label 'Ours' may denote exactly one supplied focal document-context method or
model in its source paper's result table. Reject it when no such focal technical
entity exists, when several are compatible, or when the table lacks the asserted
comparison or evaluation.
The source PAPER is also EVALUATED_ON that dataset when the table or experimental
statement reports the source paper's own empirical measurements. A bare dataset
label, cited result, or background comparison is insufficient for paper-level use.
Authorial wording that a named dataset was studied, used as an experiment, or
received reported performance measurements is sufficient for paper-level
evaluation. Model- or method-level evaluation additionally requires that exact
technical endpoint to be named or unambiguously anteceded in the bounded quote.
A model or component whose weights are explicitly initialized or transferred
from a model pretrained on a named dataset inherits TRAINED_ON that dataset as
pretraining-data provenance. Unspecified pretraining or architecture reuse
without transferred learned weights is insufficient.
A source paper's first-person introduction or discussion of its own named new
variant supports PAPER INTRODUCES variant; a background survey does not.
This includes first-person discussion of a named variant of the source's focal
method when a source-defined algorithm, equation, or result heading presents the
variant as part of the paper; the word "discuss" alone remains insufficient.
A source paper's own results, observations, theorems, or conclusions can support
PAPER INTRODUCES concept when they explicitly present that concept as a finding,
representation, or formulation established in the source, even without the word
'new'. Background knowledge, cited work, assumptions, and merely used concepts
remain insufficient.
When a source paper develops an approach and explicitly identifies a named
process or method as the approach's essential idea, construction, or operation,
the same passage separately supports PAPER USES_METHOD that process or method.
Introduction by itself does not prove operational use.
Invoking 'the X theorem' inside the source paper's own proof or derivation supports
PAPER USES_CONCEPT X; a background mention or cited proof summary does not.
A title supports a paper-to-task relation only when its wording explicitly
connects the work to the task, rather than merely naming a broad topic.
A dedicated technical section or experiment supports ADDRESSES for its source
PAPER or named method or model when the bounded quote includes both the heading
and body showing that endpoint's analysis, operation, or experiment. A heading
alone or background survey is insufficient.
Use contradicted for an opposing statement.
Use insufficient for ambiguity, implication, co-occurrence, or missing evidence.
Comparing a concept's scope in an entity does not prove that entity uses it.
Preserve grammatical roles: 'A uses B within C' supports A USES B, not C USES B.
Do not reassign a nested, comparative, or contextual noun as the predicate subject.
Return one decision for every relation_id and do not add new triples.
