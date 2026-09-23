Identify focal entities needed to interpret document-wide discourse.
Use only entity types explicitly enabled for document context.
When PAPER is enabled and the front matter explicitly identifies this source's
own title, return exactly one PAPER first; do not omit it in favor of a model
or method that competes for the bounded context inventory.
For a scholarly source, return its own paper title and any primary named model
or method that the title or abstract unambiguously presents as this source's work.
After returning the PAPER, independently audit every supplied title, abstract,
and front-matter contribution sentence for those primary technical entities; the
paper record never makes this second audit complete. Include an acronym expansion
or a descriptive primary architecture when the source explicitly identifies it
as the work's own model or method, even if it is not a proper name.
Do not return cited works, background methods, baselines, or section headings.
For another document it may be the named report, case, standard, product, or work
whose title and identity are established by the supplied front matter. For a
certificate, form, label, data sheet, or specification it is the named product,
material, substance, or batch its header describes, when that type is enabled.
A location page (page 0) names the folders the document sits in and its file; a
folder or file name may name the document's subject, such as the product, material,
batch, project, or experiment it belongs to.
Return only explicit identities grounded by a short verbatim front-matter quote.
The quote must contain the canonical name or a genuine spelling/number alias.
Do not return authors, affiliations, references, topics, methods, or datasets unless
one of those types is itself enabled for document context.
Use each type's supplied contextual surfaces only as later discourse guidance.
Return an empty list when no configured focal entity is explicit.
