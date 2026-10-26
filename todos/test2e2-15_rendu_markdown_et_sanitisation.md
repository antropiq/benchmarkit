# test2e2-15 - Rendu markdown, maths, code, et assainissement

Niveau : etat d'UI (aucune generation : tout passe par l'apercu du brouillon).
Spec : `web/e2e/15-markdown-rendering.spec.js`.

## Cas d'usage couvert

Ce que l'utilisateur tape (et ce que le modele renvoie) est rendu comme du
markdown : listes, gras, tableaux, maths, blocs de code colorises - et le HTML
dangereux n'est jamais execute.

## A lire avant d'ecrire

- `web/src/lib/markdown.js` : marked (GFM, `breaks`), `marked-katex`,
  `marked-highlight`, la regle du fence ```` ```mermaid ```` (seul un fence ferme
  devient `.mermaid-block`), `DOMPurify.sanitize` (toujours)
- `web/src/components/MarkdownStream.vue`, l'apercu
  `aria-label="Message preview (not sent yet)"` dans `WorkBenchView.vue`

## Methode (economique et deterministe)

L'apercu du brouillon est rendu par le meme `renderMarkdown` qu'une reponse : il
suffit de taper dans le textarea et d'attendre l'apercu. Une attente par cas
(l'apercu est debonce) ; aucune generation LLM.

## Tests

1. Structure : taper une liste a puces + du gras -> l'apercu contient `ul > li`
   et un `strong` ; un tableau GFM produit un `table`.
2. Maths : taper `$$E=mc^2$$` -> l'apercu contient un element `.katex`.
3. Code : taper un bloc ```` ```python ```` -> `code` avec la classe
   `hljs language-python`, et le texte du bloc est conserve intact.
4. Mermaid : un fence ```` ```mermaid ```` **non ferme** (pendant un stream) reste
   un bloc de code ; une fois le fence referme, un element `.mermaid-block`
   apparait avec son `data-src`. Ne pas tester le SVG produit (rendu asynchrone
   cote `mermaid.js`) : l'assertion s'arrete a la presence du bloc.
5. Assainissement : taper `<img src=x onerror="window.__pwned=1">` et un
   `<script>window.__pwned=1</script>` -> l'apercu ne contient ni attribut
   `onerror`, ni balise `script`, et `window.__pwned` reste `undefined` (verifier
   par `page.evaluate`, justifie : c'est l'invariant de securite).

## Pieges

- Retaper/vider le brouillon entre les cas pour que l'apercu ne melange pas deux
  contenus.
- Le HTML attendu vient du rendu, pas d'un `v-html` brut : ne pas chercher le
  texte brut dans le DOM, chercher les elements.

## Acceptation

`npx playwright test e2e/15-markdown-rendering.spec.js` au vert.
