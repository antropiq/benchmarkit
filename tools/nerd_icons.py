"""Generate the workspace tree's file-type icons.

`web/src/lib/file-icons.js` maps a file name to one Nerd Font glyph, and three
artifacts must agree with it: the two woff2 subsets in `web/src/fonts/`, and the
`unicode-range` of the "JetBrains Mono Nerd Icons" faces in `nerd-fonts.css`.
All three are generated from the tables below — nothing is hand-written, and the
code points come from the source TTF rather than from anyone's memory.

    python tools/nerd_icons.py generate   # rewrite JS + woff2 + CSS (needs fonts/)
    python tools/nerd_icons.py check      # committed artifacts up to date? (no fonts/)

`fonts/JetBrainsMono/*` is an untracked download (see .gitignore), so `generate`
is the only command that reads it; the test suite runs `check`, which compares
the generated JS against what the tool would produce *and* the subset cmaps and
CSS ranges against the code points that JS declares. A mapping entry whose glyph
is not in the subset therefore fails the suite instead of rendering a tofu box.

To add a file type: name a glyph that the source TTF actually has (see the
"Rebuild" note in nerd-fonts.css), add it to EXTENSIONS or NAMES, run
`generate`, and commit the regenerated JS and woff2 with it.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parent.parent
JS_PATH = ROOT / "web/src/lib/file-icons.js"
CSS_PATH = ROOT / "web/src/fonts/nerd-fonts.css"
FONT_DIR = ROOT / "web/src/fonts"
SUBSET_WOFF2 = {400: "Regular", 700: "Bold"}
FAMILY = "JetBrains Mono Nerd Icons"

# Glyph name -> the extensions it stands for (lowercase, no leading dot).
# Grouped by what the glyph is, so the set of icons stays reviewable at a glance;
# one glyph carrying several extensions is the point (`.ts`/`.mts`/`.cts` are the
# same thing to a reader).
EXTENSIONS: Dict[str, List[str]] = {
    # web / frontend
    "seti-javascript": ["js", "mjs", "cjs"],
    "seti-react": ["jsx", "tsx"],
    "seti-typescript": ["ts", "mts", "cts"],
    "seti-vue": ["vue"],
    "seti-svelte": ["svelte"],
    "custom-astro": ["astro"],
    "seti-html": ["html", "htm", "xhtml"],
    "seti-ejs": ["ejs"],
    "seti-jinja": ["jinja", "jinja2", "twig"],
    "seti-pug": ["pug", "jade"],
    "seti-mustache": ["mustache"],
    "dev-handlebars": ["hbs", "handlebars"],
    "seti-css": ["css"],
    "seti-sass": ["scss", "sass"],
    "seti-less": ["less"],
    "seti-stylus": ["styl", "stylus"],
    # data, config, schemas
    "dev-json": ["json", "jsonc", "json5"],
    "dev-yaml": ["yaml", "yml"],
    "custom-toml": ["toml"],
    "seti-xml": ["xml"],
    "seti-svg": ["svg"],
    "seti-config": ["ini", "cfg", "conf"],
    "seti-graphql": ["graphql", "gql"],
    "dev-mysql": ["sql"],
    "dev-postgresql": ["psql"],
    "dev-sqlite": ["db", "sqlite", "sqlite3"],
    "dev-redis": ["rdb"],
    "seti-terraform": ["tf", "tfvars"],
    # documents and plain text
    "seti-markdown": ["md", "markdown", "mdx", "rst"],
    "seti-text": ["txt", "text", "log"],
    "seti-csv": ["csv", "tsv"],
    # languages
    "seti-python": ["py", "pyi"],
    "seti-ruby": ["rb"],
    "seti-go": ["go"],
    "seti-rust": ["rs"],
    "seti-java": ["java"],
    "seti-kotlin": ["kt", "kts"],
    "seti-c": ["c", "h"],
    "seti-cpp": ["cpp", "cc", "cxx", "hpp"],
    "dev-csharp": ["cs"],
    "seti-php": ["php"],
    "seti-swift": ["swift"],
    "seti-dart": ["dart"],
    "seti-lua": ["lua"],
    "seti-perl": ["pl"],
    "dev-r": ["r"],
    "seti-scala": ["scala"],
    "seti-haskell": ["hs"],
    "seti-elixir": ["ex", "exs"],
    "dev-erlang": ["erl"],
    "seti-clojure": ["clj"],
    "seti-ocaml": ["ml"],
    "seti-julia": ["jl"],
    # shells and editor configs
    "seti-shell": ["sh", "bash", "fish"],
    "custom-zsh": ["zsh"],
    "seti-powershell": ["ps1"],
    "custom-vim": ["vim"],
    "custom-emacs": ["el"],
    # notebooks, runtimes, fonts
    "seti-notebook": ["ipynb"],
    "seti-wasm": ["wasm"],
    "seti-font": ["ttf", "otf", "woff", "woff2"],
    # media, office, archives
    "seti-image": ["png", "jpg", "jpeg", "gif", "webp", "avif", "ico", "bmp"],
    "seti-video": ["mp4", "webm", "mov"],
    "seti-audio": ["mp3", "wav", "flac"],
    "seti-pdf": ["pdf"],
    "seti-word": ["doc", "docx"],
    "fa-file_excel_o": ["xls", "xlsx"],
    "fa-file_powerpoint_o": ["ppt", "pptx"],
    "seti-zip": ["zip", "tar", "gz", "tgz", "bz2", "xz", "7z", "jar"],
    # opaque payloads the Editor refuses to render — worth marking as such
    "cod-file_binary": ["so", "dll", "exe", "bin", "class", "o", "pyc"],
}

# Exact basename -> glyph, for the names an extension cannot identify. Checked
# before the extension and before the stem, so it always wins.
NAMES: Dict[str, str] = {
    "dockerfile": "seti-docker",
    "containerfile": "seti-docker",
    "makefile": "seti-makefile",
    "gnumakefile": "seti-makefile",
    "cmakelists.txt": "dev-cmake",
    "package.json": "dev-npm",
    "package-lock.json": "seti-lock",
    "yarn.lock": "seti-lock",
    "pnpm-lock.yaml": "seti-lock",
    "bun.lockb": "seti-lock",
    "deno.json": "dev-denojs",
    "vite.config.js": "dev-vite",
    "vite.config.ts": "dev-vite",
    "tailwind.config.js": "dev-tailwindcss",
    "tailwind.config.ts": "dev-tailwindcss",
    ".github": "seti-github",
    ".gitlab-ci.yml": "seti-gitlab",
    ".gitignore": "seti-git_ignore",
    ".gitattributes": "seti-git_ignore",
    ".editorconfig": "seti-editorconfig",
    "tsconfig.json": "seti-tsconfig",
    "jsconfig.json": "seti-tsconfig",
    ".eslintrc": "dev-eslint",
    ".babelrc": "seti-babel",
    ".prettierrc": "custom-prettier",
    ".env": "seti-settings",
    ".env.local": "seti-settings",
    ".env.example": "seti-settings",
    "nginx.conf": "dev-nginx",
    "build.gradle": "dev-gradle",
    "gradlew": "dev-gradle",
    "pom.xml": "dev-maven",
    ".zshrc": "custom-zsh",
    ".zshenv": "custom-zsh",
    ".zprofile": "custom-zsh",
    ".bashrc": "seti-shell",
    ".bash_profile": "seti-shell",
    ".profile": "seti-shell",
    ".vimrc": "custom-vim",
    "init.vim": "custom-neovim",
}

# Bare stems — matched after the exact name and before any extension, so the
# same entry answers `LICENSE`, `LICENSE.md` and `License.txt`.
STEMS: Dict[str, str] = {
    "readme": "seti-markdown",
    "changelog": "seti-markdown",
    "contributing": "seti-markdown",
    "license": "seti-license",
    "copying": "seti-license",
    "notice": "seti-text",
    "authors": "seti-text",
}


# The three icons that are not about a file type: the two folder states and the
# honest fallback for anything the tables above do not recognise.
ICONS: Dict[str, str] = {
    "folder": "seti-folder",
    "folderOpen": "custom-folder_open",
    "file": "fa-file_o",
}


def used_glyphs() -> List[str]:
    out = list(ICONS.values()) + list(NAMES.values()) + list(STEMS.values()) + list(EXTENSIONS)
    return sorted(set(out))


def source_ttf(weight: str = "Regular") -> Path:
    return ROOT / f"fonts/JetBrainsMono/JetBrainsMonoNerdFont-{weight}.ttf"


def source_cmap() -> Dict[str, int]:
    """Glyph name -> code point, from the untracked source face."""
    from fontTools.ttLib import TTFont

    font = TTFont(str(source_ttf()))
    cmap = font.getBestCmap()
    if not cmap:
        raise SystemExit(f"{source_ttf()}: no character map to read")
    return {name: code for code, name in cmap.items()}


def esc(code: int) -> str:
    """The JS string literal for one code point, spelled as an escape."""
    return f"'\\u{code:04x}'"


def render_js(cmap: Dict[str, int]) -> str:
    """The mapping module, code points resolved from ``cmap``."""
    missing = [name for name in used_glyphs() if name not in cmap]
    if missing:
        raise SystemExit(
            f"not in {source_ttf()}: {', '.join(missing)}\n"
            "pick a glyph the Nerd-patched face really carries (check its cmap)."
        )

    def line(key: str, glyph: str, indent: str = "  ") -> str:
        if re.fullmatch(r"[A-Za-z_$][\w$]*", key):
            return f"{indent}{key}: {esc(cmap[glyph])}, // {glyph}"
        return f"{indent}{key!r}: {esc(cmap[glyph])}, // {glyph}"

    out: List[str] = [
        "// The workspace tree's icon per entry: a Nerd Font glyph chosen to match",
        "// what the file is. GENERATED by tools/nerd_icons.py -- edit the tables",
        "// there, run `generate`, and commit this file with the regenerated woff2",
        "// subsets and nerd-fonts.css. The `// <name>` after each escape is the",
        "// Nerd Fonts glyph name it was resolved from.",
        "//",
        "// Keyed on the *name* the tree already has, never on the server's language",
        "// id: `LANGUAGE_BY_EXTENSION` on the backend says which grammar Monaco",
        "// gets and is deliberately coarse (no grammar for a Dockerfile). An icon",
        "// answers a different question -- \"what is this?\" -- so it needs the finer",
        "// set, and the two must stay independent: a new grammar is not a new icon.",
        "//",
        "// The glyphs are PUA code points (U+E5xx-U+F1xx), rendered by the",
        "// \"JetBrains Mono Nerd Icons\" face in web/src/fonts/nerd-fonts.css, whose",
        "// unicode-range is exactly the set used below. They are not text: the tree",
        "// rows already mark every one of them aria-hidden, so a screen reader",
        "// never announces a private-use character.",
        "",
        "// Extension -> glyph. Lowercase, no leading dot.",
        "const BY_EXTENSION = {",
    ]
    for glyph in sorted(EXTENSIONS, key=lambda g: (EXTENSIONS[g][0], g)):
        for ext in sorted(EXTENSIONS[glyph]):
            out.append(line(ext, glyph))
    out += ["}", "", "// Exact basename -> glyph, for the names an extension cannot",
            "// identify (a Dockerfile has no extension; `yarn.lock` and `package.json`",
            "// deserve their own). Checked before the extension, so it wins.",
            "const BY_NAME = {"]
    for name in sorted(NAMES):
        out.append(line(name, NAMES[name]))
    out += [
        "}",
        "",
        "// Bare stem -> glyph, for the family whose name is the point: `LICENSE`,",
        "// `LICENSE.md` and `License.txt` all answer to one entry. Checked after the",
        "// exact name and before the extension, so `foo.json` never reads as json-by-stem.",
        "const BY_STEM = {",
    ]
    for stem in sorted(STEMS):
        out.append(line(stem, STEMS[stem]))
    out += ["}", ""]
    # (export key, const name) -- the const is spelled in snake caps so the export
    # block below reads like the keys it maps.
    for key, const in (
        ("folder", "ICON_FOLDER"),
        ("folderOpen", "ICON_FOLDER_OPEN"),
        ("file", "ICON_FILE"),
    ):
        out.append(f"const {const} = {esc(cmap[ICONS[key]])} // {ICONS[key]}")
    out += [
        "",
        "export const ICONS = {",
        "  folder: ICON_FOLDER,",
        "  folderOpen: ICON_FOLDER_OPEN,",
        "  file: ICON_FILE,",
        "}",
        "",
        "// The name before its first dot (`LICENSE.md` -> `LICENSE`); a dotfile keeps",
        "// its leading dot, so `.github` is not read as the stem `github`.",
        "function stemOf(name) {",
        "  const index = name.indexOf('.')",
        "  return index > 0 ? name.slice(0, index) : name",
        "}",
        "",
        "// Plain lookups on an object literal would answer for `constructor` and the",
        "// rest of Object.prototype, painting a real file with an inherited icon; the",
        "// hasOwn guard is what makes a miss a miss.",
        "function own(table, key) {",
        "  return Object.hasOwn(table, key) ? table[key] : null",
        "}",
        "",
        "// One lookup per row, in the order a reader would expect it: exact name, then",
        "// stem, then extension. A folder answers with its own state, because an open",
        "// branch with a closed-folder icon is a small lie the eye notices.",
        "export function fileIcon(name, { dir = false, open = false } = {}) {",
        "  if (dir) return open ? ICON_FOLDER_OPEN : ICON_FOLDER",
        "  const lower = String(name ?? '').toLowerCase()",
        "  const byName = own(BY_NAME, lower) ?? own(BY_STEM, stemOf(lower))",
        "  if (byName) return byName",
        "  const dot = lower.lastIndexOf('.')",
        "  if (dot > 0) {",
        "    const byExtension = own(BY_EXTENSION, lower.slice(dot + 1))",
        "    if (byExtension) return byExtension",
        "  }",
        "  return ICON_FILE",
        "}",
        "",
    ]
    return "\n".join(out)


def js_code_points(text: str) -> Set[int]:
    """Every code point an escape in the generated module declares."""
    return {int(h, 16) for h in re.findall(r"\\u([0-9a-fA-F]{4})", text)}


def ranges(code_points: Iterable[int]) -> List[str]:
    ordered = sorted(code_points)
    out: List[str] = []
    start = previous = ordered[0]
    for code in ordered[1:]:
        if code == previous + 1:
            previous = code
            continue
        out.append(_token(start, previous))
        start = previous = code
    out.append(_token(start, previous))
    return out


def _token(start: int, end: int) -> str:
    return f"U+{start:04X}" if start == end else f"U+{start:04X}-{end:04X}"


def expand(tokens: str) -> Set[int]:
    out: Set[int] = set()
    for token in tokens.replace("\n", " ").split(","):
        body = token.strip()[2:]
        if not body:
            continue
        if "-" in body:
            lo, hi = body.split("-")
            out.update(range(int(lo, 16), int(hi, 16) + 1))
        else:
            out.add(int(body, 16))
    return out


def css_blocks(text: str) -> List[str]:
    return re.findall(
        rf'font-family:\s*"{re.escape(FAMILY)}";.*?unicode-range:([^;]+);', text, flags=re.S
    )


def write_css(code_points: Set[int]) -> None:
    text = CSS_PATH.read_text(encoding="utf-8")
    lines: List[str] = []
    current = ""
    for token in ranges(code_points):
        if len(current) + len(token) + 1 > 66:
            lines.append(current)
            current = token
        else:
            current = f"{current},{token}" if current else token
    lines.append(current)
    block = ",\n    ".join(lines)
    pattern = re.compile(
        rf'(font-family:\s*"{re.escape(FAMILY)}";.*?unicode-range:\s*)([^;]+)(;)', flags=re.S
    )
    updated, count = pattern.subn(lambda m: f"{m.group(1)}{block}{m.group(3)}", text)
    if count != len(SUBSET_WOFF2):
        raise SystemExit(f"{CSS_PATH}: rewrote {count} faces, expected {len(SUBSET_WOFF2)}")
    CSS_PATH.write_text(updated, encoding="utf-8")


def write_subset(cmap: Dict[str, int], weight: int, source_weight: str) -> None:
    """Subset one source face down to exactly the glyphs the mapping uses."""
    unicodes = ",".join(f"U+{cmap[name]:04X}" for name in used_glyphs())
    out = FONT_DIR / f"nerd-icons-{weight}.woff2"
    subprocess.run(
        [
            sys.executable, "-m", "fontTools.subset",
            f"--unicodes={unicodes}", "--flavor=woff2", "--no-hinting",
            "--desubroutinize", str(source_ttf(source_weight)),
            f"--output-file={out}",
        ],
        check=True,
    )


def generate() -> int:
    """Rewrite the JS module, both woff2 subsets and the CSS from the tables."""
    cmap = source_cmap()
    rendered = render_js(cmap)
    JS_PATH.write_text(rendered, encoding="utf-8")
    code_points = js_code_points(rendered)
    for weight, source_weight in SUBSET_WOFF2.items():
        write_subset(cmap, weight, source_weight)
    write_css(code_points)
    print(f"{len(used_glyphs())} glyphs, {len(code_points)} code points written")
    return 0


def check() -> int:
    """Are the committed artifacts exactly what the tables would produce?"""
    from fontTools.ttLib import TTFont

    problems: List[str] = []
    if not source_ttf().exists():
        # The generator's output is the contract. Without the untracked download
        # we cannot re-render the JS, so verify the shipped artifacts against the
        # code points the committed module itself declares.
        declared = js_code_points(JS_PATH.read_text(encoding="utf-8"))
    else:
        # With the download present the whole module is re-renderable, so a
        # hand-edit of it is caught here rather than only via the font cmaps.
        rendered = render_js(source_cmap())
        if JS_PATH.read_text(encoding="utf-8") != rendered:
            problems.append(
                f"{JS_PATH}: does not match the tables -- regenerate"
            )
        declared = js_code_points(rendered)

    for weight in SUBSET_WOFF2:
        path = FONT_DIR / f"nerd-icons-{weight}.woff2"
        cmap = TTFont(str(path)).getBestCmap()
        if cmap is None:
            raise SystemExit(f"{path}: no character map to read")
        shipped = set(cmap)
        for label, diff in (
            ("is missing", declared - shipped),
            ("carries unused", shipped - declared),
        ):
            if diff:
                problems.append(
                    f"nerd-icons-{weight}.woff2 {label}: "
                    + ", ".join(sorted(f"U+{c:04X}" for c in diff))
                )
    blocks = css_blocks(CSS_PATH.read_text(encoding="utf-8"))
    if len(blocks) != len(SUBSET_WOFF2):
        problems.append(f"CSS: {len(blocks)} {FAMILY!r} faces, expected {len(SUBSET_WOFF2)}")
    for block in blocks:
        if expand(block) != declared:
            problems.append("CSS: a unicode-range does not match the used code points")
            break

    for message in problems:
        print(message)
    if problems:
        print("run: python tools/nerd_icons.py generate")
        return 1
    print(f"{len(declared)} code points: module, subsets and @font-face agree")
    return 0


COMMANDS = {"generate": generate, "check": check}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f"usage: {Path(__file__).name} {'|'.join(COMMANDS)}")
    raise SystemExit(COMMANDS[sys.argv[1]]())
