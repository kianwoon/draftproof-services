"""Internal modules for the DraftProof rewrite pipeline.

The public compatibility entrypoint remains ``poc.rewrite_pipeline``.  These
modules are extracted behind that facade so production imports and legacy tests
can migrate gradually without behavior changes.
"""
