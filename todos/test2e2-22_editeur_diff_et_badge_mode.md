# test2e2-22 - Editeur : onglet diff, badge de mode, survie aux changements d'onglet

Niveau : etat d'UI (le serveur shared sert `/api/fs/diff`).
Spec : `web/e2e/22-editor-diff-mode.spec.js`.

## Cas d'usage couvert

L'utilisateur regarde ce qu'un fichier est devenu depuis le dernier commit, et
l'app lui dit honnetement si le panneau est editable ou non. Son travail (onglets
ouverts, arbre developpe) survit a un coup d'oeil au Benchmark.

## A lire avant d'ecrire

- `web/src/components/WorkspaceNavigator.vue` : le menu contextuel (`role=menu`,
  l'item "View diff…" n'apparait que si l'entree git est `comparable: true`)
- `web/src/composables/useGitStatus.js` (le champ `comparable`, `isDiffable`) et
  `web/src/composables/useEditorFiles.js` (`openDiff`, cle `diff:<path>`)
- `web/src/components/EditorArea.vue` : la note `editor-raw-note` ("left: as of
  the last commit - right: as it is on disk"), l'`aria-label="Diff of <path>"`
- `web/src/composables/useEditorMode.js` : le badge `read-write` seulement quand
  le WorkBench est idle ; un onglet diff est toujours `read-only`
- `App.vue` : l'onglet Editor reste monte (`v-show`), donc onglets et terminal
  survivent a un changement d'onglet

## Tests

1. Menu contextuel (etat stubbe) : `page.route('/api/fs/status')` renvoie une
   entree `{path: 'README.md', kind: 'modified', comparable: true}` -> clic droit
   sur la ligne `README.md` -> `role=menu` visible avec l'item "View diff…". Une
   entree `comparable: false` ne propose pas l'item (le fichier n'a pas de cote
   commité).
2. Ouverture du diff : cliquer "View diff…" -> un second onglet s'ouvre pour le
   meme fichier, `GET /api/fs/diff?path=README.md` a repondu 200, la note
   expliquant les deux colonnes est visible, et le badge du panneau passe a
   `read-only`.
3. Deux onglets pour un chemin : l'onglet fichier et l'onglet diff coexistent
   (deux chips, clefs differentes) ; le badge redevient `read-write` quand
   l'onglet fichier redevient actif.
4. Badge de mode pendant une reponse : lancer une reponse de chat depuis
   WorkBench, passer sur l'onglet Editor pendant le stream -> le badge lit
   `read-only` et son `title` explique que le WorkBench edite peut-etre ces
   fichiers. (1 seule generation pour la spec.)
5. Survie a un changement d'onglet : avec un fichier ouvert et une branche
   developpee, aller sur Benchmark puis revenir -> la chip est toujours la, la
   branche est toujours developpee (aucune nouvelle requete `GET /api/fs/file`).

## Pieges

- Le clic droit doit viser la ligne de l'arbre (`.tree-row`), pas son libelle :
   le menu se place au curseur et se limite aux fichiers.
- `page.route` sur `/api/fs/status` est la seule facon deterministe d'avoir un
   fichier "diffable" : le depot reel change d'etat entre les specs.

## Acceptation

`npx playwright test e2e/22-editor-diff-mode.spec.js` au vert.
