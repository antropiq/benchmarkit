// One place to say what the two files in the workspace's reserved folder are
// doing (pol-05's read-write policy, am-03's agent-modes file).
//
// The server reports `{state, path, error}` for each of them on both
// working-location endpoints. Surfaces want a sentence about each — the settings
// dialog after a switch, the same dialog on a plain read, and (from pol-07) the
// Editor banner — and they must not each invent one, because the *tone* carries
// meaning: `failed` is a warning that the switch succeeded and the seed did not,
// `absent` is a plain fact, `created`/`existing` is reassurance that the
// project's own config is readable and editable outside the app.

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

// The agent-modes file (am-03) answers with the same four states, and its
// sentences follow the same tone rule — but its `absent` case is less dire than
// the policy's: a workspace with no modes file still gets the built-in modes in
// the dropdown, so the sentence says what is missing rather than what breaks.
// Only `failed` is a `warn`. Its text is a complete sentence that repeats the
// "working location is set" preamble rather than depending on the policy note
// above it: the two files fail independently, and a modes-only failure with no
// policy failure would otherwise be a line that refers to nothing. The note
// never goes into `pathLocationError` — the location moved and both hubs
// restarted, and a red line there would claim the switch itself had not.
export function describeAgentModes(agentModes) {
  if (!agentModes || !agentModes.state) return null
  switch (agentModes.state) {
    case 'created':
      return {
        tone: 'plain',
        text: `Created this project's agent-modes file: ${agentModes.path}. It is edited outside the app, and it decides what the WorkBench modes dropdown offers.`,
      }
    case 'existing':
      return {
        tone: 'plain',
        text: `This project has its own agent-modes file: ${agentModes.path}.`,
      }
    case 'absent':
      return {
        tone: 'plain',
        text: 'This project has no agent-modes file, so the modes dropdown shows benchmarkit\u2019s built-in modes.',
      }
    case 'failed':
      return {
        tone: 'warn',
        text: `Working location is set, but benchmarkit could not create the agent-modes file (${agentModes.path ?? '?'}): ${
          agentModes.error || 'unknown error'
        }. The modes dropdown uses benchmarkit\u2019s built-in modes.`,
      }
    default:
      return null
  }
}
