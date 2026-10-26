import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const WEB_ROOT = dirname(dirname(fileURLToPath(import.meta.url)))

// FastAPI only mounts the SPA when `web/dist` exists (see the static mount in
// src/api/server.py), so a e2e run against a stale or absent build silently
// tests the previous frontend. Build unless dist is there and the run was
// told to trust it (E2E_SKIP_BUILD=1, for iterating on specs while `npm run
// build -- --watch` feeds dist).
export default async function globalSetup() {
  if (existsSync(join(WEB_ROOT, 'dist')) && process.env.E2E_SKIP_BUILD === '1') {
    return
  }
  execFileSync('npm', ['run', 'build'], { cwd: WEB_ROOT, stdio: 'inherit' })
}
