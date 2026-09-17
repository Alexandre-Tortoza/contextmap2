## Summary

Describe the change and why it is needed.

## Type

- [ ] Feature
- [ ] Fix
- [ ] Research
- [ ] Experiment
- [ ] Refactor
- [ ] Tests / benchmark
- [ ] Documentation
- [ ] CI / maintenance

## Validation

Describe how the change was validated. Include metrics, benchmark outputs, fixtures, or generated artifacts when map quality may be affected.

## Impact on generated artifacts

- [ ] No artifact/schema change
- [ ] Compatible artifact/schema change
- [ ] Incompatible artifact/schema change

If the artifact changes, describe the affected fields, provenance, migration expectations, and validation performed.

## Checklist

- [ ] `make check` passes locally
- [ ] Tests cover the relevant behavior
- [ ] Experimental behavior is isolated from the validated path
- [ ] Documentation reflects externally visible changes
- [ ] No dataset-specific assumption was introduced without being documented
