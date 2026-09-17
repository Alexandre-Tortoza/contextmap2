# Repository settings

This document records the intended GitHub repository settings so they can be audited alongside the source code.

## Repository metadata

Description:

```text
Persistent 3D contextual map generation from synchronized robotic sensor data.
```

Suggested topics:

```text
robotics
3d-mapping
semantic-mapping
lidar
computer-vision
ros2
open-vocabulary
vision-language-models
scene-graphs
research
```

## Merge policy

Recommended repository settings:

- default branch: `main`;
- allow squash merge: enabled;
- allow rebase merge: enabled;
- allow merge commits: disabled;
- automatically delete head branches after merge: enabled;
- allow auto-merge: optional after required checks are configured.

## Branch policy

### `main`

- changes arrive through pull requests;
- expected source branch is `qa`;
- require CI before merge;
- require conversation resolution;
- block force pushes;
- block branch deletion.

### `qa`

- changes arrive through pull requests;
- expected source branch is `dev`;
- require CI before merge;
- block force pushes;
- block branch deletion.

### `dev`

- integration branch for short-lived development and research branches;
- require CI before merge when branch rules are enabled;
- block force pushes where practical.

The `Branch policy` GitHub Action also validates the expected promotion path for pull requests.

## Required checks

Once GitHub branch rules are enabled, the minimum required checks should include:

```text
CI / quality
Branch policy / validate
```

CodeQL should remain enabled for `main`, but it does not need to block early research changes unless the repository security policy requires it.

## Releases

Only `main` should be tagged for release. Release tags follow `vMAJOR.MINOR.PATCH`; `v0.x.y` remains the validation series.

## Repository security

Recommended GitHub features:

- Dependabot alerts and security updates;
- secret scanning where available;
- private vulnerability reporting;
- CodeQL scanning;
- branch rules for `main` and `qa`.
