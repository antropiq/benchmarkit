// One place to say what the workspace policy file is doing (pol-05).
//
// The server reports `{state, path, error}` for `<working location>/.nojae/read-write-policy.json`
// on both working-location endpoints. Three surfaces want a sentence about it —
// the settings dialog after a switch, the same dialog on a plain read, and (from
// pol-07) the Editor banner — and they must not each invent one, because the
// *tone* carries meaning: `failed` is a warning that the switch succeeded and
// the seed did not, `absent` is a plain fact, `created`/`existing` is reassurance
// that the project's rules are readable and editable outside the app.

export const POLICY_STATES = ['created', 'existing', 'absent', 'failed']

// `tone` is what a caller renders it as: 'warn' is a line the user should read
// but not act on, 'plain' is context.
export function describePolicy(policy) {
  if (!policy || !policy.state) return null
  switch (policy.state) {
    case 'created':
      return {
        tone: 'plain',
        text: `Created this project's read-write policy: ${policy.path}. It is edited outside the app, and it decides what the Editor hides or locks.`,
      }
    case 'existing':
      return {
        tone: 'plain',
        text: `This project has its own read-write policy: ${policy.path}.`,
      }
    case 'absent':
      return {
        tone: 'plain',
        text: 'This project has no read-write policy file, so its tree is not filtered by any workspace rules.',
      }
    case 'failed':
      // Deliberately NOT an error line: the location moved and both hubs
      // restarted, so the switch succeeded. Only the file could not be written.
      return {
        tone: 'warn',
        text: `Working location is set, but benchmarkit could not create its policy file (${policy.path ?? '?'}): ${
          policy.error || 'unknown error'
        }. That project's tree is not filtered by any workspace rules yet.`,
      }
    default:
      return null
  }
}
