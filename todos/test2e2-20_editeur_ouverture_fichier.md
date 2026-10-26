# test2e2-20 - Editeur : ouverture d'un fichier, onglets, refus de politique

Niveau : etat d'UI (le serveur shared sert `/api/fs/file`, lecture seule).
Spec : `web/e2e/20-editor-open-file.spec.js`.

## Cas d'usage couvert

L'utilisateur clique un fichier : son contenu s'affiche dans un onglet, on peut
l'ouvrir une deuxieme fois sans doublon, le fermer ; un fichier protege est
refuse avec une raison lisible.

## A lire avant d'ecrire

- `web/src/composables/useEditorFiles.js` : `openFile` (append-or-focus, `binary`
  -> message d'info sans onglet, `state.message` avec le chemin et le ton),
  `closeTab`, `focusTab`
- `web/src/components/EditorArea.vue` : `role=tablist[aria-label="Open files"]`,
  les chips (`.editor-tab-name` avec `role=tab`, `aria-selected`), le bouton
  `aria-label="Close <name>"`, le panneau `role=tabpanel`, l'etat vide "Open a
  file", le bandeau `role=alert` + son bouton "Dismiss this message"
- `src/core/editorfs/policy.py` + `src/core/editorfs/constants.py`
  (`MAX_EDITOR_BYTES`, le sniff binaire) et le payload de `GET /api/fs/file`
  (`path`, `text`, `mtime`, `language`, `size`, `binary?`, `truncated?`)

## Tests

1. Ouverture depuis l'arbre : ouvrir l'onglet Editor, developper `src/api`,
   cliquer la ligne `events.py` -> une chip nommee `events.py` apparait avec
   `aria-selected=true`, un `GET /api/fs/file?path=src/api/events.py` a repondu
   200 (verifier la reponse de la requete ecouteee : sa taille est non nulle), et
   le panneau `role=tabpanel` heberge la surface editee.
2. Append-or-focus : cliquer la meme ligne une seconde fois -> une seule chip pour
   ce chemin et **aucun** second `GET /api/fs/file` (le cache fait foi).
3. Deux fichiers, deux onglets : ouvrir un deuxieme fichier -> deux chips, l'onglet
   actif est le second, un clic sur la premiere chip la reselectionne
   (`aria-selected`) sans nouvelle requete.
4. Fermeture : le bouton `aria-label="Close events.py"` retire la chip ; apres
   fermeture de la derniere, l'etat vide "Open a file" revient.
5. Fichier protege (etat stubbe) : `page.route('/api/fs/file')` repond 422 avec un
   `detail` -> un bandeau `role=alert` affiche la raison **et le chemin**, sans
   creer de chip ; son bouton "Dismiss this message" le retire. Justifier le stub :
   la politique est deja testee cote serveur, ce test verifie le rendu du refus.
6. Binaire (etat stubbe) : reponse `{binary: true, size: 1234}` -> un message
   d'information "binary file, …" et **aucune** chip.

## Pieges

- La 1re ouverture de l'onglet charge le chunk Monaco : attendre la presence de
   l'editeur plutot qu'un delai.
- L'editeur est en lecture seule (aucune route d'ecriture) : ne tenter aucune
   modification du contenu dans le panneau.

## Acceptation

`npx playwright test e2e/20-editor-open-file.spec.js` au vert.
