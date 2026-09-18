import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { appEnv } from "@/server/env";

export async function POST(request: Request) {
  const form = await request.formData();
  const files = form.getAll("files").filter((value): value is File => value instanceof File);
  if (files.length === 0) {
    return Response.json({ error: "No files uploaded" }, { status: 400 });
  }
  const jobId = String(form.get("jobId") || randomUUID());
  const directory = path.join(appEnv().stateRoot, "uploads", jobId);
  await mkdir(directory, { recursive: true });
  for (const file of files) {
    const bytes = Buffer.from(await file.arrayBuffer());
    await writeFile(path.join(directory, file.name.replaceAll("/", "_")), bytes);
  }
  return Response.json({ path: directory, count: files.length, jobId });
}
