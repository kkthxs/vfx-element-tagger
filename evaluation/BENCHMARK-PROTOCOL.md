# VFX analysis validation

The eight existing artist examples are development constraints, not an independent test set.
Repeatedly tuning against them cannot establish general accuracy. Saved human-corrected results
must not be counted as fresh machine predictions.

## Held-out annotation

1. Prepare a model-output-free queue with `python3 scripts/prepare_benchmark.py --output evaluation/heldout-annotations.json`.
2. Recruit artists to annotate at least 100-200 independent elements before inspecting predictions.
   The current catalog may contain too few eligible groups; the tool reports the actual shortfall.
3. Keep proxy/master variants, near duplicates and source families in one split. Do not train or
   tune prompts on held-out groups. Include black, alpha, key-screen and photographed plates;
   sparse particles; short flashes; multiple takes and simultaneous instances; mixed materials;
   edge crops; very dark elements; empty clips and unsupported media.
4. Label visible primary/secondary materials, performer action, shot scale/viewpoint, edge contact,
   temporal arc, screen/depth motion, counts and key-event intervals. Record unknowns and disputes
   explicitly. A specific lens focal length is not visual ground truth without camera metadata.
5. Have a second artist review disagreements, particularly bolt/branch counts and edge contact.
   Maintain label revisions separately from predictions. Do not silently change labels to agree
   with a new model.

## Evaluation

Run fresh predictions into an isolated catalog with generic filename context, no human overrides
and no inherited facets. Record source fingerprints, code revision, model identities, actual input
frame indices/timestamps/resolution, errors and wall time. Use per-field constraints rather than
one optimistic all-text keyword score. `range_covers` checks that proposed trims retain labeled
event intervals; it does not establish that the trim is editorially optimal.

Report labeled and missing asset counts, per-field results, accepted versus review-required
coverage, false confident accepts, parse/decode failures and runtime. Treat model confidence as
uncalibrated. Missing assets count as failures, not as an improved denominator. Partial human
corrections establish only those fields, not the accuracy of the rest of the description.

Retrieval needs separate labeled queries with relevant and irrelevant assets, filename-blind
variants, Recall@5/MRR, and hard negative precision. A query finding an asset by filename does not
prove that the visual model understood it. The current concept ranker has no installed semantic
embedding index; do not describe its hash vectors as learned embeddings.

The six-clip stress suite and synthetic regressions are diagnostic safeguards. Passing them is
necessary but insufficient for production-quality semantic accuracy.
