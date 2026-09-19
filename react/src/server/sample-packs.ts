import { stat } from "node:fs/promises";
import path from "node:path";

/** A hosted corpus the console can point a first run at, when it is on this host. */
export interface SamplePack {
  name: string;
  /** Relative to the repository root, as the pipeline's `files.input_path`. */
  path: string;
  graphName: string;
  why: string;
}

const SAMPLE_PACKS: readonly SamplePack[] = [
  {
    name: "Martial arts",
    path: "data/martial_arts/files",
    graphName: "Martial arts history",
    why: "10 markdown files, gold.json beside them",
  },
  {
    name: "Deep learning papers",
    path: "data/deep_learning_papers/files",
    graphName: "Deep learning papers",
    why: "PDF-heavy sample with gold relations",
  },
];

/**
 * The packs whose files are actually here.
 *
 * A laptop checkout has both; the fleet image ships the small one and the
 * papers must be downloaded first. Offering a pack that is not on the host
 * only produces "Input path does not exist" after the click.
 */
export async function availableSamplePacks(repositoryRoot: string): Promise<SamplePack[]> {
  const present = await Promise.all(
    SAMPLE_PACKS.map(async (pack) => {
      try {
        return (await stat(path.join(repositoryRoot, pack.path))).isDirectory();
      } catch {
        return false;
      }
    }),
  );
  return SAMPLE_PACKS.filter((_, index) => present[index]);
}
