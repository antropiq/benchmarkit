// One place to say what the files in the workspace's reserved folder are doing
// (pol-05's read-write policy, am-03's agent-modes file, mbl-02's prompt-helper
// file).
//
// The server reports `{state, path, error}` for each of them on both
// working-location endpoints, and the app bar's working-location field is the
// one surface that speaks about them. It renders only the `warn` sentences, as
// the one line under its field, and stays silent on the `plain` ones — the bar
// has room for a warning, not for a caption. It gets its sentences from here
// rather than inventing them, because the *tone* carries meaning:
// `failed` is a warning that the switch succeeded and the seed did not, `absent`
// is a plain fact, `created` is reassurance that the project's own config is
// readable and editable outside the app. The helper file is not that kind of
// config: it is the app's own store (spec D1), so its sentences describe what
// the bullets above the composer became, not who may edit the file — and it
// answers two states of its own, `migrated` and `migration-skipped` (spec D6).

export const POLICY_STATES = ['created', 'existing', 'absent', 'failed']

// The two extra states only the prompt-helpers file answers (spec D6's lift of
// the old app-level file), on top of the shared four above.
export const PROMPT_HELPER_STATES = [...POLICY_STATES, 'migrated', 'migration-skipped']

// `tone` is what a caller renders it as: 'warn' is a line the user should read
// but not act on, 'plain' is context.
//
// A state the user has nothing to learn from answers `null`: `existing` is the
// ordinary case of a project that has had its file since the last switch, and
// naming the path of a file nobody asked about is noise, so the three describers
// say nothing about it. `absent`, `failed` and the rest still get their sentence.
export function describePolicy(policy) {
  if (!policy || !policy.state) return null
  switch (policy.state) {
    case 'created':
      return {
        tone: 'plain',
        text: `Created this project's read-write policy: ${policy.path}. It is edited outside the app, and it decides what the Editor hides or locks.`,
      }
    case 'existing':
      return null
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
      return null
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

// The third file of the reserved folder (mbl-02): the nine prompt helpers
// behind the bullets above the composer. It answers the four shared states in
// the same shape, plus two of its own from the one-time lift of the old
// app-level file (spec D6) — `migrated` is good news and stays `plain`, while
// `migration-skipped` is a `warn` because the user has to act on it (fix the
// old file's entry count, or fill the new bullets by hand). As with the other
// two, `failed` repeats the "working location is set" preamble: the three seeds
// fail independently, and a helpers-only failure must not read as a line about
// a note that was never shown.
export function describePromptHelpers(promptHelpers) {
  if (!promptHelpers || !promptHelpers.state) return null
  switch (promptHelpers.state) {
    case 'created':
      return {
        tone: 'plain',
        text: `Created this project's prompt-helper file: ${promptHelpers.path}. The bullets above the WorkBench textarea are now this project's own.`,
      }
    case 'existing':
      return null
    case 'absent':
      return {
        tone: 'plain',
        text: 'This project has no prompt-helper file yet, so the bullets above the WorkBench textarea are empty until one is seeded or saved.',
      }
    case 'failed':
      return {
        tone: 'warn',
        text: `Working location is set, but benchmarkit could not create the prompt-helper file (${promptHelpers.path ?? '?'}): ${
          promptHelpers.error || 'unknown error'
        }. The bullets above the WorkBench textarea are empty.`,
      }
    case 'migrated':
      return {
        tone: 'plain',
        text: `benchmarkit moved this machine's older bullet set into ${promptHelpers.path}: the prompts that used to be app-wide (from resources/memory_registration.json) are now this project's.`,
      }
    case 'migration-skipped':
      return {
        tone: 'warn',
        text: `Working location is set, but benchmarkit could not move this machine's older bullet set into ${
          promptHelpers.path ?? '?'
        }: ${
          promptHelpers.error || 'the legacy file could not be read'
        }. This project starts with empty bullets, and the older file was left where it is.`,
      }
    default:
      return null
  }
}
