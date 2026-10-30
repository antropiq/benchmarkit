import { describe, expect, it } from 'vitest'

import { scopeFor } from '../src/lib/search-scope'

// ---------------------------------------------------------------------------
// The row the search gesture was made on -> the scope the dialog names.
//
// Three scopes and one order: the root row answers `dir: true`, so a rule that
// asks about `dir` first reads "the whole workspace" as "a folder" and searches
// one subtree instead of everything. Nothing else here is clever enough to be
// worth a test — except that the *kind* comes from the row's `dir` flag and
// never from what the name looks like.
// ---------------------------------------------------------------------------

const rootLabel = () => 'benchmarkit'

describe('scopeFor', () => {
  it('names the whole workspace for the root row, not an empty label', () => {
    const scope = scopeFor({ path: '', name: 'benchmarkit', dir: true, parent: '' }, rootLabel)
    expect(scope).toEqual({ kind: 'workspace', label: 'benchmarkit', path: '', isFile: false })
    expect(scope.label).not.toBe('')
  })

  it('names a folder row by its own path', () => {
    const scope = scopeFor({ path: 'docs/old', name: 'old', dir: true, parent: 'docs' }, rootLabel)
    expect(scope).toEqual({ kind: 'folder', label: 'docs/old', path: 'docs/old', isFile: false })
  })

  it('names a file row by its own path, and isFile is true only there', () => {
    const scope = scopeFor(
      { path: 'src/config.py', name: 'config.py', dir: false, parent: 'src' },
      rootLabel,
    )
    expect(scope).toEqual({
      kind: 'file',
      label: 'src/config.py',
      path: 'src/config.py',
      isFile: true,
    })
  })

  // The row's `dir` flag is the answer; a name is not evidence of anything.
  it('reads dir from the row, not from what the name looks like', () => {
    expect(
      scopeFor({ path: 'index.html', name: 'index.html', dir: true, parent: '' }, rootLabel).kind,
    ).toBe('folder')
    const licence = scopeFor({ path: 'LICENSE', name: 'LICENSE', dir: false, parent: '' }, rootLabel)
    expect(licence.kind).toBe('file')
    expect(licence.isFile).toBe(true)
  })

  it('declines when there is no row', () => {
    expect(scopeFor(null, rootLabel)).toBeNull()
    expect(scopeFor(undefined, rootLabel)).toBeNull()
    // A target whose path was never filled in is the same case: the command has
    // no subject, and guessing the workspace root from it would search more
    // than the user asked for.
    expect(scopeFor({ name: 'notes.md', dir: false }, rootLabel)).toBeNull()
  })

  it('leaves the label of the workspace case to rootLabel, called with no argument', () => {
    const seen = []
    scopeFor({ path: '', name: '', dir: true, parent: '' }, (...args) => {
      seen.push(args)
      return 'ws'
    })
    expect(seen).toEqual([[]])
  })
})
