import { describe, expect, it } from 'vitest'

import { isMarkdownPath } from '../src/lib/markdown-preview'

// Layer 1: the pure predicate the tree menu reads *before* any file is read.
// It has to agree with the server's own extension table
// (`LANGUAGE_BY_EXTENSION` in src/core/editorfs/files.py), because a menu item
// must never promise a preview for a file the pane labels `plaintext`.

describe('isMarkdownPath', () => {
  it('accepts a markdown file at any depth', () => {
    expect(isMarkdownPath('README.md')).toBe(true)
    expect(isMarkdownPath('a/b.MD')).toBe(true)
    expect(isMarkdownPath('docs/notes.md')).toBe(true)
    // A dot inside a folder name does not move the extension.
    expect(isMarkdownPath('v1.2/CHANGELOG.md')).toBe(true)
  })

  it('refuses an extension the server does not call markdown', () => {
    // There is no `.markdown` entry in LANGUAGE_BY_EXTENSION, so the pane would
    // open this as `plaintext` — the menu must not offer Preview for it.
    expect(isMarkdownPath('a.markdown')).toBe(false)
    expect(isMarkdownPath('a.mdx')).toBe(false)
    expect(isMarkdownPath('notes.md.json')).toBe(false)
    expect(isMarkdownPath('src/a.py')).toBe(false)
    expect(isMarkdownPath('md')).toBe(false)
  })

  it('treats a dotfile as having no extension, like language_for() does', () => {
    expect(isMarkdownPath('.md')).toBe(false)
    expect(isMarkdownPath('dir/.md')).toBe(false)
    expect(isMarkdownPath('.gitignore')).toBe(false)
  })

  it('has an answer for a path that is not a string', () => {
    expect(isMarkdownPath('')).toBe(false)
    expect(isMarkdownPath(null)).toBe(false)
    expect(isMarkdownPath(undefined)).toBe(false)
  })
})
