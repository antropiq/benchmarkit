# Live smoke checklist — the workspace-scoped model connection

The human-only half of this feature's verification (spec
`nojae-settings-workspace-spec.md` §10, last paragraph). Everything the suites can
prove hermetically they already prove against a throw-away `BENCHMARKIT_PATH_LOCATION`
and a fake hub; what is listed here is the one thing they cannot — a real switch
between two real workspaces, a real language server coming up on it, and a real
restart picking up a hand-edit.

**No task in this lot runs this list, and none ever will.** It exists so the human
step is not invented later.

## The rule that binds it

Run every item **against this machine's own servers** (the endpoint this install is
already configured with — do not add a remote one for the occasion), and in **a
scratch workspace, never a project that matters**: the switch seeds a `.nojae`
folder into whatever folder it moves to, and the dialog writes into it.

Create two scratch folders for it — call them **A** and **B** — and leave the app
on A to start.

**Only item 5 needs a restart**, by design: a hand-edit of the settings file is not
hot (spec D4), exactly as a hand-edit of `.env` never was. Items 1–4 are all
observable in one session.

## The five observations

1. **Workspace A answers a chat.** Open A as the working location, open Settings,
   set the model (server URL, key if needed, model id), save, and send one chat
   message from WorkBench.
   *Expected:* the answer streams. The Settings hint at the top of the dialog names
   `A/.nojae/nojae-settings.json`, and the dialog says saving applies right away
   while a hand-edit needs a server restart.

2. **Switch to B in the app bar.** Type B's path into the working-location field and
   press Set.
   *Expected:* the app bar moves to B and the switch answers without an error. B is
   brand new, so its `.nojae` holds the six seeded files, `nojae-settings.json`
   among them.

3. **The dialog shows B, not A.** Reopen Settings without reloading anything.
   *Expected:* the hint names **`B/.nojae/nojae-settings.json`** — a different path
   from item 1 — and the Server URL field holds B's own values, which for a freshly
   seeded workspace are the template's instruction sentences. The not-configured
   note is visible under the LLM section, naming B's path. **No chat request dials
   those sentences**: sending one refuses with that same sentence, and the sentence
   never appears as a transport error quoting the template's text.

4. **B's enabled language server is live without a reboot.** Before the switch, write
   a `B/.nojae/lsp-settings.json` declaring one server with `"enable": true`, and
   list that id in `lspEnabled` in `B/.nojae/nojae-settings.json` (both by hand,
   before the switch — the file is invisible to the Editor by design).
   *Expected:* right after the switch, the LSP section of Settings shows that server
   `running`/`active` — the switch seeded the settings file *before* restarting the
   hub (spec D10), so the workspace's own `lspEnabled` was already readable when the
   hub asked. A server that only comes up after pressing *Restart* means the
   ordering regressed.

5. **A hand-edit needs a restart, and then it answers.** Switch back to A, edit
   `A/.nojae/nojae-settings.json` by hand (change the model id), and reopen Settings
   **without** restarting the app.
   *Expected:* the dialog still shows the **old** value — a hand-edit is not hot
   (D4), and that is the behaviour, not a bug. Restart the server, reopen Settings:
   the new value is there, and a chat turn goes to that model.

## What this list does *not* cover

- A malformed file: `tests/test_workspace_settings_seed.py` asserts the refusal for
  both a chat and a benchmark request, and `tests/test_workspace_settings.py` the
  reader's states.
- The seed's ordering as a mechanism: `tests/test_workspace_settings_seed.py` pins
  it with a recording hub, including the case where the target was browsed first.
- The dialog following the workspace: `web/e2e/03-workspace-switch.spec.js` runs the
  first half of items 1–3 against a real server and a real scratch root.

So a failure here that no test reproduces is worth a test of its own — but write it
against a `tmp_path` workspace, never against the machine's own `.nojae`.
