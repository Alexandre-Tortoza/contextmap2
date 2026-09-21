"""On-disk serialization of the ContextMap: the portable ContextMapArtifact.

The schema modules of ``contextmap.artifact`` define what a map *is* and perform no file I/O.
This subpackage decides how a map is stored, written, read, validated and exported. Nothing here
is a public entry point: the public surface is the root of ``contextmap.artifact``. See
``src/contextmap/artifact/docs/storage-layout.md``.
"""
