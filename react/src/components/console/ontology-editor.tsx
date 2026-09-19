"use client";

import { useState } from "react";
import { X } from "lucide-react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import type { OntologySelection, OntologyTerm } from "@/server/protocol/schema";
import { cn } from "@/lib/utils";

/** How a name must read to be a type: PERSON, FOUNDED_BY. */
const TERM_NAME = /^[A-Z][A-Z0-9_]{1,31}$/;

export const RELATION_MODES: Array<{ id: OntologySelection["relations"]; label: string; hint: string }> = [
  {
    id: "guided",
    label: "Guided by this list",
    hint: "The extractor prefers these predicates and may add one the text plainly states.",
  },
  { id: "fixed", label: "Only this list", hint: "Every relation is one of these; anything else is dropped." },
  {
    id: "open",
    label: "Detected automatically",
    hint: "No list. The extractor names each relation from the text (LOCATED_IN, TRAINED_UNDER…).",
  },
];

/** Turn what someone typed into a type name, or nothing. */
export function normalizeTermName(raw: string): string | null {
  const name = raw
    .trim()
    .replace(/[\s-]+/g, "_")
    .replace(/[^A-Za-z0-9_]/g, "")
    .toUpperCase();
  return TERM_NAME.test(name) ? name : null;
}

/** Why a selection cannot be built with, or null when it can. */
export function ontologyProblem(selection: OntologySelection): string | null {
  if (selection.entityTypes.length === 0) {
    return "Add at least one entity type.";
  }
  if (selection.relations !== "open" && selection.relationTypes.length === 0) {
    return "Add at least one relation type, or let relations be detected automatically.";
  }
  return null;
}

export function OntologyEditor({
  value,
  onChange,
  defaults,
  locked = null,
}: {
  value: OntologySelection;
  onChange: (next: OntologySelection) => void;
  /** What the runtime extracts unless changed here, and where that comes from. */
  defaults: { selection: OntologySelection; source: string } | null;
  /** When set, the vocabulary is shown but not editable, for this reason. */
  locked?: string | null;
}) {
  const [intent, setIntent] = useState("");
  const propose = trpc.ingestion.ontology.useMutation({
    onError: (error) => toast.error(error.message),
  });
  const proposal = propose.data;
  const problem = ontologyProblem(value);
  const differsFromDefault =
    defaults !== null && JSON.stringify(defaults.selection) !== JSON.stringify(value);

  function proposedTerms(names: string[]): OntologyTerm[] {
    return names.map((name) => ({ name, description: proposal?.descriptions[name] ?? "" }));
  }

  function merge(existing: readonly OntologyTerm[], added: OntologyTerm[]): OntologyTerm[] {
    const seen = new Set(existing.map((term) => term.name));
    return [...existing, ...added.filter((term) => !seen.has(term.name) && seen.add(term.name))];
  }

  return (
    <Card data-testid="ontology-editor">
      <CardHeader>
        <CardTitle>What to extract</CardTitle>
        <CardDescription>
          {locked
            ? locked
            : "Entity types name what the extractor looks for; relations say how they connect. They apply to this graph only. Describe the graph for a suggestion, or edit the lists directly."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {locked ? null : (
          <div className="space-y-2">
            <Textarea
              aria-label="Describe the graph you want"
              placeholder="e.g. people, schools, and techniques in these martial-arts histories"
              value={intent}
              onChange={(event) => setIntent(event.target.value)}
              rows={2}
            />
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                onClick={() => propose.mutate({ intent })}
                disabled={intent.trim().length < 8 || propose.isPending}
              >
                {propose.isPending ? "Suggesting…" : "Suggest types"}
              </Button>
              {proposal ? (
                <>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() =>
                      onChange({
                        ...value,
                        entityTypes: proposedTerms(proposal.types),
                        relationTypes: proposedTerms(proposal.relations),
                        relations: value.relations === "open" ? "guided" : value.relations,
                      })
                    }
                  >
                    Replace with suggestion
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      onChange({
                        ...value,
                        entityTypes: merge(value.entityTypes, proposedTerms(proposal.types)),
                        relationTypes: merge(value.relationTypes, proposedTerms(proposal.relations)),
                      })
                    }
                  >
                    Add to mine
                  </Button>
                </>
              ) : null}
            </div>
            {proposal ? (
              <div className="space-y-1 rounded-md border border-border bg-muted/30 p-3 text-sm" data-testid="ontology-proposal">
                <p className="text-muted-foreground">
                  {proposal.source === "model"
                    ? "Suggested by the model from your description."
                    : proposal.modelFailure
                      ? `The model did not answer (${proposal.modelFailure}), so these are the nouns of your description.`
                      : "No model is configured for the console, so these are the nouns of your description."}
                </p>
                <TermRow label="Types" terms={proposedTerms(proposal.types)} />
                <TermRow label="Relations" terms={proposedTerms(proposal.relations)} />
              </div>
            ) : null}
          </div>
        )}

        <TermListEditor
          label="Entity types"
          testId="entity-types"
          terms={value.entityTypes}
          placeholder="Add a type, e.g. TECHNIQUE"
          locked={Boolean(locked)}
          onChange={(entityTypes) => onChange({ ...value, entityTypes })}
        />

        <div className="space-y-2">
          <p className="text-sm font-medium">Relations</p>
          <div className="grid gap-2 md:grid-cols-3" role="radiogroup" aria-label="Relation typing">
            {RELATION_MODES.map((mode) => (
              <button
                key={mode.id}
                type="button"
                role="radio"
                aria-checked={value.relations === mode.id}
                disabled={Boolean(locked)}
                onClick={() => onChange({ ...value, relations: mode.id })}
                className={cn(
                  "flex flex-col items-start gap-1 rounded-lg border px-3 py-2 text-left transition-colors disabled:cursor-default disabled:opacity-70",
                  value.relations === mode.id
                    ? "border-primary bg-accent ring-2 ring-primary"
                    : "border-border bg-background hover:bg-muted/50",
                )}
              >
                <span className="text-sm font-medium">{mode.label}</span>
                <span className="text-xs leading-snug text-muted-foreground">{mode.hint}</span>
              </button>
            ))}
          </div>
          {value.relations === "open" ? null : (
            <TermListEditor
              label="Relation types"
              testId="relation-types"
              terms={value.relationTypes}
              placeholder="Add a relation, e.g. TRAINED_UNDER"
              locked={Boolean(locked)}
              onChange={(relationTypes) => onChange({ ...value, relationTypes })}
            />
          )}
        </div>

        {problem && !locked ? <Alert variant="destructive">{problem}</Alert> : null}
        {defaults && !locked ? (
          <p className="text-xs text-muted-foreground">
            {differsFromDefault ? (
              <>
                Changed from {defaults.source}.{" "}
                <button type="button" className="underline" onClick={() => onChange(defaults.selection)}>
                  Reset to {defaults.source}
                </button>
              </>
            ) : (
              <>These are the types from {defaults.source}.</>
            )}
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}

function TermRow({ label, terms }: { label: string; terms: OntologyTerm[] }) {
  return (
    <p>
      <span className="font-medium">{label}: </span>
      {terms.length ? (
        terms.map((term, index) => (
          <span key={term.name} title={term.description || undefined}>
            <span className="font-mono text-xs">{term.name}</span>
            {index < terms.length - 1 ? ", " : ""}
          </span>
        ))
      ) : (
        <span className="text-muted-foreground">none</span>
      )}
    </p>
  );
}

function TermListEditor({
  label,
  testId,
  terms,
  placeholder,
  locked,
  onChange,
}: {
  label: string;
  testId: string;
  terms: readonly OntologyTerm[];
  placeholder: string;
  locked: boolean;
  onChange: (terms: OntologyTerm[]) => void;
}) {
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [description, setDescription] = useState("");

  function add() {
    const name = normalizeTermName(draft);
    if (!name) {
      toast.error("A type name is letters, digits and underscores, like TECHNIQUE or TRAINED_UNDER.");
      return;
    }
    if (terms.some((term) => term.name === name)) {
      toast.error(`${name} is already listed.`);
      return;
    }
    onChange([...terms, { name, description: "" }]);
    setDraft("");
  }

  const current = editing ? terms.find((term) => term.name === editing) : undefined;

  return (
    <div className="space-y-2" data-testid={testId}>
      <p className="text-sm font-medium">{label}</p>
      <div className="flex flex-wrap gap-1.5">
        {terms.length === 0 ? <p className="text-xs text-muted-foreground">None yet.</p> : null}
        {terms.map((term) => (
          <span
            key={term.name}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-xs",
              editing === term.name ? "border-primary bg-accent" : "border-border bg-background",
            )}
          >
            <button
              type="button"
              title={term.description || "No description yet - click to add one"}
              className="disabled:cursor-default"
              disabled={locked}
              onClick={() => {
                setEditing(editing === term.name ? null : term.name);
                setDescription(term.description);
              }}
            >
              {term.name}
            </button>
            {locked ? null : (
              <button
                type="button"
                aria-label={`Remove ${term.name}`}
                className="rounded-sm text-muted-foreground hover:text-foreground"
                onClick={() => {
                  onChange(terms.filter((item) => item.name !== term.name));
                  if (editing === term.name) {
                    setEditing(null);
                  }
                }}
              >
                <X className="size-3" aria-hidden="true" />
              </button>
            )}
          </span>
        ))}
      </div>
      {current && !locked ? (
        <div className="flex flex-col gap-2 rounded-md border border-border bg-muted/30 p-3 sm:flex-row sm:items-center">
          <span className="font-mono text-xs">{current.name}</span>
          <Input
            aria-label={`Description of ${current.name}`}
            className="h-8 flex-1"
            placeholder="What qualifies, in one sentence"
            value={description}
            onChange={(event) => setDescription(event.target.value)}
          />
          <Button
            size="sm"
            variant="secondary"
            onClick={() => {
              onChange(terms.map((term) => (term.name === current.name ? { ...term, description: description.trim() } : term)));
              setEditing(null);
            }}
          >
            Save
          </Button>
        </div>
      ) : null}
      {locked ? null : (
        <div className="flex gap-2">
          <Input
            aria-label={`Add ${label.toLowerCase().replace(/s$/, "")}`}
            className="h-8 max-w-xs"
            placeholder={placeholder}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                add();
              }
            }}
          />
          <Button size="sm" variant="outline" onClick={add} disabled={!draft.trim()}>
            Add
          </Button>
        </div>
      )}
    </div>
  );
}
