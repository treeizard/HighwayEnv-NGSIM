# Static map assets

These are the canonical checked-in previews for the US-101 and Japanese map
topologies. Runtime environments do not load these PNG files.

`examples/highway/render_map.py` may generate a new `us101_static.png` in the
caller's chosen output location; its default output name is not a dependency
on a repository-root image.

The former root-level copies were byte-identical and were moved to the
parent-project cleanup quarantine on 2026-07-26. Their checksums and restore
manifest are recorded in the parent repository's redundancy register and
research diary.
