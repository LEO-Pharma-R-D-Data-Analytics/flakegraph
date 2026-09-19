"use client";

import { useMemo, useState } from "react";
import { Check, ChevronDown, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { communityMemberIds } from "@/server/graph-filter";
import { cn } from "@/lib/utils";

export interface Neighborhood {
  id: string;
  title: string;
  /** Entities the community names; the list ranks by it. */
  size: number;
}

/** The communities of a dataset as pickable neighborhoods, largest first. */
export function neighborhoodsOf(communities: readonly Record<string, unknown>[]): Neighborhood[] {
  return communities
    .map((community) => ({
      id: String(community.id ?? ""),
      title: String(community.title ?? community.id ?? ""),
      size: communityMemberIds(community).length,
    }))
    .filter((item) => item.id)
    .sort((left, right) => right.size - left.size || left.title.localeCompare(right.title));
}

/**
 * Choose neighborhoods to focus the graph on.
 *
 * A graph has tens to hundreds of neighborhoods, so the row shows only the
 * chosen ones as removable chips; the rest sit in a panel with a search
 * and a ranked list, where a title reads on one line beside its size.
 */
export function NeighborhoodPicker({
  neighborhoods,
  value,
  onChange,
  trailing,
}: {
  neighborhoods: readonly Neighborhood[];
  value: readonly string[];
  onChange: (next: string[]) => void;
  /** Controls that belong on the same row, after the chips. */
  trailing?: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const byId = useMemo(() => new Map(neighborhoods.map((item) => [item.id, item])), [neighborhoods]);
  const chosen = value.map((id) => byId.get(id)).filter((item): item is Neighborhood => Boolean(item));
  const matching = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return needle ? neighborhoods.filter((item) => item.title.toLowerCase().includes(needle)) : neighborhoods;
  }, [neighborhoods, search]);

  function toggle(id: string) {
    onChange(value.includes(id) ? value.filter((item) => item !== id) : [...value, id]);
  }

  return (
    <div className="space-y-2" data-testid="neighborhood-picker">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
          Neighborhoods
          <span className="ml-1.5 normal-case tracking-normal">{neighborhoods.length.toLocaleString()}</span>
        </p>
        <Button
          size="sm"
          variant="outline"
          className="h-7 gap-1 px-2 text-xs"
          aria-expanded={open}
          aria-controls="neighborhood-list"
          onClick={() => setOpen((current) => !current)}
        >
          {chosen.length ? `${chosen.length} chosen` : "Choose neighborhoods"}
          <ChevronDown className={cn("size-3.5 transition-transform", open ? "rotate-180" : "")} aria-hidden="true" />
        </Button>
        {chosen.map((item) => (
          <span
            key={item.id}
            className="inline-flex h-7 max-w-xs items-center gap-1 rounded-md border border-primary/40 bg-accent pl-2 pr-1 text-xs"
          >
            <span className="truncate" title={item.title}>
              {item.title}
            </span>
            <button
              type="button"
              aria-label={`Remove ${item.title}`}
              className="rounded-sm p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground"
              onClick={() => toggle(item.id)}
            >
              <X className="size-3" aria-hidden="true" />
            </button>
          </span>
        ))}
        {chosen.length > 1 ? (
          <Button size="sm" variant="ghost" className="h-7 px-2 text-xs" onClick={() => onChange([])}>
            Clear
          </Button>
        ) : null}
        {trailing}
      </div>
      {open ? (
        <div id="neighborhood-list" className="rounded-md border border-border bg-background">
          <div className="border-b border-border p-2">
            <Input
              aria-label="Find a neighborhood"
              className="h-8"
              placeholder="Find a neighborhood"
              value={search}
              autoFocus
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
          <ul className="max-h-72 overflow-y-auto py-1" role="listbox" aria-label="Neighborhoods" aria-multiselectable="true">
            {matching.length === 0 ? (
              <li className="px-3 py-2 text-sm text-muted-foreground">No neighborhood matches.</li>
            ) : (
              matching.map((item) => {
                const selected = value.includes(item.id);
                return (
                  <li key={item.id} role="option" aria-selected={selected}>
                    <button
                      type="button"
                      className={cn(
                        "flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-muted/60",
                        selected ? "bg-accent/60" : "",
                      )}
                      onClick={() => toggle(item.id)}
                    >
                      <span
                        className={cn(
                          "flex size-4 shrink-0 items-center justify-center rounded-sm border",
                          selected ? "border-primary bg-primary text-primary-foreground" : "border-border",
                        )}
                        aria-hidden="true"
                      >
                        {selected ? <Check className="size-3" /> : null}
                      </span>
                      <span className="min-w-0 flex-1 truncate" title={item.title}>
                        {item.title}
                      </span>
                      <span className="shrink-0 tabular-nums text-xs text-muted-foreground">
                        {item.size.toLocaleString()} {item.size === 1 ? "entity" : "entities"}
                      </span>
                    </button>
                  </li>
                );
              })
            )}
          </ul>
          <div className="flex items-center justify-between border-t border-border px-3 py-1.5 text-xs text-muted-foreground">
            <span>
              {search.trim() ? `${matching.length.toLocaleString()} of ` : ""}
              {neighborhoods.length.toLocaleString()} neighborhoods · {chosen.length} chosen
            </span>
            <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={() => setOpen(false)}>
              Done
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
