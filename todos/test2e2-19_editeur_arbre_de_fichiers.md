# test2e2-19 - Editeur : arbre du dossier de travail

Niveau : etat d'UI (le serveur shared sert `/api/fs/*` sans hub).
Spec : `web/e2e/19-editor-tree.spec.js`.

## Cas d'usage couvert

L'utilisateur ouvre l'onglet Editor et navigue dans le projet : un niveau a la
fois, les dossiers ignores sont absents, ce que git a modifie est marque.

## A lire avant d'ecrire

- `web/src/components/WorkspaceNavigator.vue` (ligne racine, bouton "Refresh the
  file tree", bouton "Close all opened folders in the file tree", le pied de page
  sur la politique, le menu contextuel `role=menu`)
- `web/src/components/WorkspaceTreeItem.vue` (row `role=treeitem`, `data-path`,
  marqueur git `title` selon le kind)
- `web/src/composables/useWorkspaceTree.js` (une liste par niveau, cache,
  `reveal`, `collapseAll`, `handleKey`)
- `src/core/editorfs/constants.py` (`IGNORED_DIR_NAMES` : `.git`, `node_modules`,
  `__pycache__`, `dist`, `.venv`, `.pytest_cache`) et `src/core/editorfs/policy.py`
- `GET /api/fs/tree?path=`, `GET /api/fs/status` (une fois pour tout l'arbre)

## Tests

1. Ouverture de l'onglet : cliquer l'onglet `Editor` -> l'onglet devient
   selectionne, `aside[aria-label="Workspace files"]` et `role=tree` sont visibles,
   le badge `read-write` / `read-only` de l'en-tete est present (le WorkBench
   etant idle : `read-write`).
2. Une liste a la fois : la ligne racine est `aria-expanded=true` et ses enfants
   sont des `role=treeitem`. Developper `src` (clic sur sa ligne) provoque
   exactement un `GET /api/fs/tree?path=src` (ecouter les requetes) ; ses enfants
   apparaissent avec `aria-level` plus grand.
3. Dossiers ignores : aucun `data-path` ne contient `.git`, `node_modules`,
   `__pycache__` ou `web/dist`.
4. Pliage global : le bouton "Close all opened folders in the file tree" est
   `disabled` quand seul le niveau racine est ouvert, actif apres avoir
   developpe `src`, et le re-plie ; les lignes de `src` disparaissent.
5. Marqueurs git (etat stubbe, le depot reel n'est pas deterministe) :
   `page.route('/api/fs/status')` renvoie `{repo: true, entries: [{path:
   'README.md', kind: 'modified'}]}` -> la ligne `README.md` porte un marqueur au
   `title` "modified since the last commit", et la ligne racine un marqueur
   "contains modified or untracked files". `GET /api/fs/status` n'est appele
   qu'une fois a l'ouverture, pas par ligne.
6. Clavier : `ArrowDown` change la ligne portant le `tabindex="0"` ; `ArrowLeft`
   sur la racine developpee la replie. Aucun `GET /api/fs/file` n'est emis.

## Pieges

- L'onglet Editor charge un chunk lazy (Monaco/xterm) : la 1re ouverture peut
  prendre une seconde, attendre l'element plutot que le clic.
- Ne pas ecrire dans l'arbre : l'onglet est en lecture seule (decision D4).

## Acceptation

`npx playwright test e2e/19-editor-tree.spec.js` au vert.
