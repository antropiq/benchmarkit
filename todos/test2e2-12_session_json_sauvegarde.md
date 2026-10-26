# test2e2-12 - Session WorkBench : sauvegarde et rechargement du JSON

Niveau : etat d'UI + une seule generation reelle.
Spec : `web/e2e/12-chat-session.spec.js`.

## Cas d'usage couvert

L'utilisateur met une conversation de cote dans un fichier JSON, puis la reouvre
plus tard pour continuer ou la rejouer ailleurs.

## A lire avant d'ecrire

- `web/src/components/WorkBenchInput.vue` : `saveSession()` (un lien avec
  l'attribut `download` telecharge le JSON), `pickSessionFile()` (input file
  `accept="application/json,.json"`, cache), `onSessionFilePicked` + le
  `window.confirm` de remplacement
- `useChat.js` : `conversationJson()`, `parseSessionFile()`, `loadSession()`
- Format du fichier : `{format, version, messages:[{role, content, reasoning?,
  images?}]}`

## Tests

1. Telechargement : faire un tour de chat (1 generation), cliquer
   `aria-label="Save session as JSON"`, capter l'evenement `download` de
   Playwright, lire le contenu (`download.path()` + `fs.readFile`) : le JSON a les
   cles attendues, `messages` contient le tour, les roles sont `user` /
   `assistant` dans l'ordre. Le nom du fichier commence par `chat-session-`.
2. Rechargement sur une conversation vide : apres `Restart conversation`,
   `setInputFiles` sur l'input file de session (le cibler par son `accept`, il est
   masque) avec le fichier enregistre -> les bulles reapparaissent (bon nombre de
   messages utilisateur et assistant) sans aucun nouvel appel a l'API.
3. Confirmation de remplacement : avec une conversation deja peuplee, charger un
   fichier -> une boite de confirmation native apparait ; la refuser (`dialog`
   dismiss) laisse la conversation intacte.
4. Fichier invalide : charger un JSON mal forme -> le message d'erreur de
   WorkBench (`.error-line`) commence par "Cannot load session:" et la
   conversation n'a pas change.

## Pieges

- L'input de session est masque (`aria-hidden`) : cibler l'element `input` par son
  `accept`, et `setInputFiles` directement (ne pas passer par le clic).
- Le `window.confirm` est natif : `page.once('dialog', d => d.dismiss())` avant
  l'action.
- Le fichier de sortie vient d'un telechargement navigateur : le deplacer vers un
  dossier de test, et ne pas ecrire dans le depot.

## Acceptation

`npx playwright test e2e/12-chat-session.spec.js` au vert.
