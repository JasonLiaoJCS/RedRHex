# RedRHex Training Panel Changelog

This changelog is for humans who would rather train robots than decode commit archaeology.

Format:

- `Added`: new capability.
- `Changed`: behavior changed.
- `Fixed`: bug fixed.
- `Notes`: operational reminders.

## Unreleased / Next

Nothing yet. Nice and suspiciously calm.

## V3.4 First Release

### Added

- Mother-side training queue:
  - New training requests become `queued` when Isaac/GPU actions are active.
  - Queued runs start automatically when the GPU is free.
  - History cards show `queued` and provide `Cancel Queue`.

### Changed

- Isaac/GPU lock now includes training, play, video recording, and ONNX export.
- Play/video/ONNX launch buttons are disabled while training is active.

### Fixed

- Prevented rapid training launches from opening overlapping Isaac sessions and exhausting GPU memory.
- Improved History attribution so failed runs do not borrow checkpoints from nearby successful runs.
- Faster `panel_...` and `wheg...` merge during active training.

## V3.2.1 Terrain Stack

### Added

- Terrain preset workflow for Mother and Child.
- Terrain diff indicators in History.
- Run-scoped terrain replay for Play and Record Video when saved terrain metadata is available.
- Robot-follow camera support for panel-launched playback/video.

### Changed

- Reward and terrain preset selection is used directly for training.
- Removed extra “Use for training” confirmation flow.
- Child History folder browsing became folder-first.

### Fixed

- Reduced confusing duplicate display between Mother-launched `panel_...` runs and discovered RSL-RL `wheg...` folders.
- Improved terminal/console labels: launch command vs output.
- Replaced less-useful attach-copy emphasis with command-copy emphasis in console flows.

## V3.2.0 Remote Refinement

### Added

- Requester-scoped Discord notification settings in Child.
- Team Activity mission-control view in Mother.
- Activity leaderboard, action mix, outcome mix, team pulse, and member-collapsed logs.
- Child dark mode and welcome message.

### Changed

- Child menu order became phone-first: Train first, Dashboard last.
- Child History gained reward comparison visibility.
- Child video labels became clearer about checkpoint iteration.

### Fixed

- Dark mode menu contrast issues.
- Child History metadata layout that mixed unrelated facts too tightly.

## V3.1.x Child Phone UX

### Added

- Phone-first inline History details.
- Folder-first Child History navigation.
- Back-to-folders control on phone and desktop.
- Bigger reward editing controls on phone.
- Smooth expand/collapse behavior for selected History cards.

### Changed

- Dashboard simplified to operational status instead of full diagnostics.
- Detailed health checks moved to Connection.

### Fixed

- History scroll jumping during auto-refresh/actions.
- Reward tab mobile width issues.
- Video playback reset during refresh.

## V3.0 Remote Team System

### Added

- `RedRHex To Go` static GitHub Pages app.
- Supabase-backed login, roles, jobs, runs, artifacts, events, and machine heartbeat.
- Remote worker command:

```bash
python -m tools.training_panel.remote_worker
```

- Control Center for worker lifecycle:
  - start
  - stop
  - restart
  - tmux/child mode
  - auto-start
  - accept/pause remote jobs
- Private Supabase Storage video playback through signed URLs.
- Discord/email notification scaffolding.
- Cloudflare Tunnel support for live services.

### Changed

- Mother became the authoritative admin/debug/control surface.
- Child became the simplified team-facing interface.

### Notes

- Machine secrets must stay on the training PC.
- Public child page may contain only Supabase URL and anon/publishable key.

## V2.x Remote Foundations

### Added

- Supabase schema for profiles, machines, runs, jobs, events, artifacts, proxy sessions, and notifications.
- Worker heartbeat and job claim flow.
- Remote role model:
  - viewer
  - operator
  - admin
- Artifact syncing foundation.

### Fixed

- Better Supabase setup diagnostics.
- Clearer missing environment variable errors.
- Correct handling of Supabase `/rest/v1` URL duplication.

## V1.1 Local Power Tools

### Added

- `Export ONNX` from History.
- `Compact Run` to keep the newest checkpoint and delete old top-level `model_*.pt`.
- ONNX metadata in History.
- Compact preview and exact run-id confirmation.

### Changed

- History Actions panel became more organized.

## V1.0 Local Training Panel

### Added

- Local Mother panel.
- Train form.
- History list.
- Process Console.
- TensorBoard launch.
- Play checkpoint.
- Record Video.
- Notes and folder organization.
- Reward preset workflow.
- Version and team identity markings.

### Notes

- This is where the panel became useful enough to stop living only in terminal commands.

## Changelog Rules For Future Us

When adding a feature:

1. Add a short entry here.
2. Mention user-facing behavior, not only filenames.
3. Put operational gotchas under `Notes`.
4. Keep it compact. Future us has training runs to babysit.
