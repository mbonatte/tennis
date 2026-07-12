# Result schema 1.0

`result.json` is a stable object with `schema_version`, `pipeline_version`, `estimates`, `input_filename`, relative `output_video`/`result_json` names, probed `metadata`, submitted `analysis_options` and `visualization_options`, arrays `shots`, `bounces`, `scene_cuts`, `points`, `plots`, dictionaries `player_statistics` and `summary`, plus `warnings`.

Paths are always relative to the job output and public API responses never include server paths. A shot may contain frame, player role, speed, and heuristic reason. A classified bounce may contain frame, phase, in/out call, court region, projected coordinates, and rule. Model-free bounce entries use `classification: unclassified`. Nullable estimates serialize as JSON `null`. Additive fields may appear in 1.x; incompatible changes require a new schema major version.
