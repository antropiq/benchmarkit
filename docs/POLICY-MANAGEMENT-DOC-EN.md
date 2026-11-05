# Workspace policy — a user's guide

How benchmarkit decides which files of a project it may show, read and change,
and how you set that up for **your** project.

Nothing here is about benchmarkit's internals: it is about one small JSON file
inside the folder you point the app at.

---

## 1. The idea in one paragraph

The **working location** is the folder the Editor tab browses and edits.
Benchmarkit asks one question about every file in it: *may I show it, read it,
change it?* The answer comes from a file you own:

```text
<working location>/.nojae/read-write-policy.json
```

Three answers are possible, called *modes*:

| mode | in the file tree | can be opened / read | can be changed (save, create inside, rename, delete) |
|---|---|---|---|
| `rw` | yes | yes | yes — this is the default for anything no rule mentions |
| `ro` | yes | yes | **no** |
| `hidden` | **no** | **no** — it looks as if it does not exist | **no** |

The policy also applies to the file tools the Chat assistant uses: a folder you
mark `ro` is not editable by an agent either, and a `hidden` folder is invisible
to it. One file, one answer, whoever is asking.

## 2. Where the file comes from

When a folder becomes your working location — at app start, or when you change it
in the **app bar** — benchmarkit writes a ready-made policy file there **if
none exists**. That default hides the usual noise (`.git`, caches) and makes
dependency and build folders read-only, so you start from a worked example rather
than from an empty file.

Two things to know about that seeding:

* it **never overwrites** a file you edited — an existing policy is left alone;
* if the folder cannot take the file (read-only mount, owned by someone else),
  the app tells you and still opens the project; you can create the file by hand.

Benchmarkit does not change this file afterwards, and there is no interface in
the app that edits it. **You edit it outside the app**, in the editor of your
choice, and the new rules are in force from the next browse — no restart needed.

The folder `.nojae` is reserved by benchmarkit: it never appears in the file tree
and no route can read or write it. That is why the file is edited outside the app
and why no rule in it may name `.nojae`.

It is also **machine-local**: `.nojae/` is not under source control, so each
checkout of a project gets its own. Keep it out of git on purpose — the sensible
rules for a laptop checkout and a server checkout are not the same.

## 3. The file

```json
{
  "version": 1,
  "readme": [
    "benchmarkit reads this file when it browses this folder.",
    "Rules are matched in order and the LAST match wins."
  ],
  "policies": [
    { "path": ".git",                         "mode": "hidden" },
    { "path": "**/__pycache__",               "mode": "hidden" },
    { "path": "**/node_modules",              "mode": "ro",
      "reason": "browse the docs, edit nothing" },
    { "path": "**/node_modules/**/README.md", "mode": "rw" }
  ]
}
```

| key | required | what it is |
|---|---|---|
| `version` | yes | must be `1` |
| `readme` | no | text for you. JSON has no comments, so this is where notes live. It is never matched |
| `policies` | yes | the list of rules, in the order they apply |

A rule has three fields:

| field | required | what it is |
|---|---|---|
| `path` | yes | the pattern, relative to the working location |
| `mode` | yes | `hidden`, `ro` or `rw` |
| `reason` | no | your own words. They are shown back to you (and to the assistant) whenever that rule is what stopped something |

`reason` is worth using: a refusal that says *"browse the docs, edit nothing"*
explains itself months later, while *"ro"* does not.

## 4. Patterns — what actually matches

Rules are matched against the path **inside the working location**, so the file
`web/src/api/server.js` in `/home/you/proj` is `web/src/api/server.js` — never
the absolute path, never a path with the project name in front.

| spelling | means |
|---|---|
| `web` | that folder — **and everything inside it** |
| `web/src` | that folder and its whole subtree |
| `*.md` | any `.md` file at the top level only |
| `**/*.md` | any `.md` file anywhere |
| `config?` | `configa`, `config1` — `?` is one character |
| `.env*` | names *starting* with `.env`: `.env`, `.env.local` |
| `**/node_modules` | `node_modules` at the top level **or at any depth** |
| `web/**/dist` | any `dist` folder under `web`, at any depth |

Four rules that catch most people out:

1. **`*` never crosses a directory.** `web/*` covers `web/index.html` but **not**
   `web/src/api/server.js`. To cover a tree, write the folder name alone (`web`)
   or use `web/**`.
2. **A rule naming a folder covers its whole subtree.** `.git` is enough to hide
   `.git/config`, `.git/objects/…` and everything else below.
3. **Names are case-sensitive.** `dist` and `Dist` are two different folders on
   the level of the rules: write the name exactly as the folder spells it, and
   on a case-insensitive filesystem pick one spelling and keep to it.
4. **A rule cannot leave the workspace.** Absolute paths (`/etc`), `~`, `..` and
   a trailing slash are refused as errors — you cannot protect a file that is
   outside the folder you pointed the app at, and you do not need to: the working
   location already is that border.

## 5. Order matters: the last match wins

Rules are read top to bottom and the **last** one that matches decides. That is
how you carve an exception out of a big rule:

```json
{ "path": "**/node_modules",              "mode": "ro" },
{ "path": "**/node_modules/**/README.md", "mode": "rw" }
```

The second line only works because it comes **after** the first. Move it above
and it does nothing.

The one exception is `hidden`: a hidden folder is cut off entirely, so nothing
below it can be brought back. A carve-out written inside a hidden folder is
ignored — a later `ro` or `rw` rule cannot rescue anything from under a `hidden`
one — and benchmarkit tells you so as a *note* rather than an error. To hide most
of a folder while keeping it browsable, use `ro` on the folder and `hidden` on
what you want gone:

```json
{ "path": "secrets",            "mode": "ro" },
{ "path": "secrets/tokens",     "mode": "hidden" },
{ "path": "secrets/README.md",  "mode": "rw" }
```

## 6. Recipes

**Keep a vendor folder browsable but untouchable**

```json
{ "path": "third_party", "mode": "ro" }
```

**Keep a build tree browsable, but hide the bulk of its output**

```json
{ "path": "**/build",        "mode": "ro" },
{ "path": "**/build/assets", "mode": "hidden" }
```

**Protect a whole area, then hand back one file**

```json
{ "path": "db/migrations",           "mode": "ro" },
{ "path": "db/migrations/README.md", "mode": "rw" }
```

**Lock generated code that always comes back from a generator**

```json
{ "path": "src/generated/**", "mode": "ro",
  "reason": "regenerated by make codegen, hand edits are lost" }
```

**Guard a directory that does not exist yet**

```json
{ "path": ".aws", "mode": "hidden" }
```

A rule about an absent folder is legal and simply waits for it to appear.

## 7. When something is wrong

There are two kinds of complaint, and they behave differently.

**Errors** — the file is not understood: a rule with a typo, a wrong `mode`
spelling, `"policies"` written `"policyies"`, a missing `version`, broken JSON.
Then the file contributes **no rules at all**, and **nothing in the workspace can
be changed** until it is fixed — not because the rules are strict, but because
the rules you wrote to protect a folder demonstrably did not load. Reading and
browsing stay available so you can see what is wrong. The message names the file
and the rule.

**Notes** — the file is fine, one rule will simply never be heard. A note is
information, not a wall; the rest of the file works.

Common things you will see, what they mean and what to do:

| what benchmarkit says | what it means | what to do |
|---|---|---|
| *the workspace policy did not load, so nothing in this workspace can be changed until it does* | one error anywhere in the file disables every write in the project | fix that one line; the file is named in full |
| *is not valid JSON: … (line N, column M)* | a comma, a quote, a trailing comma | open the file at that line |
| *has an unknown key "policis"* | a top-level key is misspelled, so your rules were never read | fix the spelling — a key nobody reads is not the same as no key |
| *mode "readonly" is not one of "hidden", "ro", "rw"* | the mode must be spelled exactly | use `ro` |
| *policies[3] path "web/node_module" can never match: `web` has no "node_module" — did you mean "web/node_modules"?* | a typo in a folder name | take the suggestion; `[3]` is the position of the rule in the `policies` list, counting from 0 |
| *path ".nojae" names the folder this file lives in, which benchmarkit reserves* | no rule may address the folder holding the rules | delete that rule |
| *it is a bare "\*\*", which would apply to the whole workspace* | a rule cannot cover the workspace root itself | write `**/*`, or name the folders |
| *it is absolute: a rule is written relative to the workspace* | patterns never start with `/` | drop the leading path |
| *policies[5] sits inside the hidden rule at policies[1]* | a note: that carve-out cannot revive anything inside a hidden folder | change the parent rule to `ro`, or remove the carve-out |
| *policies[2] repeats policies[0]* | a note: two identical rules, only the later one is ever quoted | keep one |

Index numbers in those messages count the entries of the `policies` list from
**0**, in the order they are written.

## 8. Gotchas, gathered

1. **A broken file costs you the whole project, not one folder.** One bad comma
   makes every save, create, rename and delete in the workspace refuse to run.
   After editing the file, try a harmless save (or just browse): if the tree
   still answers and writes work, the file is understood.
2. **The rules are not deleted, they are ignored** — for as long as the file does
   not load. Nothing looks different in the tree, which is what makes an "edits
   are off" message easy to miss.
3. **`*` is not `**`.** The single most common mistake, and it hides or locks far
   more than intended.
4. **A folder rule is a subtree rule.** `{"path": "web", "mode": "hidden"}` hides
   the whole frontend. There is no way to hide a folder while keeping its
   contents, and that is by design.
5. **Carving out of a hidden folder is a no-op.** Use `ro` on the parent (§5).
6. **Later wins — except for `hidden`.** When two rules of different strength
   overlap, the one further down the list answers; but a `hidden` match always
   beats a looser one wherever it sits. Broad rules first, exceptions after.
7. **A rule for a folder that is not there yet is not an error** — but a rule one
   or two letters away from a real folder name **is**. If benchmarkit suggests a
   sibling, it is almost always right.
8. **You cannot unlock what benchmarkit fixed.** Inside benchmarkit's own folder,
   a handful of paths are protected by the app itself; a workspace rule trying to
   loosen one earns a note and no effect. In any other project this file is the
   whole law, so such a note never appears — and there a file like `.env` is an
   ordinary file: hide it yourself if you want it hidden.
9. **The file is per checkout.** Copying `.nojae/read-write-policy.json` to a new
   clone is a manual step; a fresh clone gets the default file instead.
10. **`hidden` is not a secret store.** It hides a file from the Editor's tree,
    from its reads and from the assistant's file tools. It does not encrypt
    anything: the file is still on disk, still in git, readable by every program
    on the machine.
11. **Deleting a default rule has an effect.** Those lines are why `node_modules`
    is browsable-but-locked and `.git` is invisible; drop one and that protection
    is gone from the next browse.
12. **Editing the file takes effect immediately.** Reload the tree; there is no
    restart to do, and the app will not rewrite the file behind you.

## 9. Checklist for a new project

1. Point the working location at the project (the app bar's location field).
2. Open `<project>/.nojae/read-write-policy.json` in an external editor.
3. Read the default rules; delete what does not fit this project.
4. Add `ro` for anything derived or vendored that you still want to read.
5. Add `hidden` for anything you never want to see or touch.
6. Add a `reason` to any rule you expect to wonder about in six months.
7. Save, then browse the tree: the shape you see is the policy you wrote.
