# test2e2-21 - Editeur : le shell du terminal integre

Niveau : session reelle (une WebSocket + un PTY, sans LLM).
Spec : `web/e2e/21-editor-terminal.spec.js`.

## Cas d'usage couvert

L'utilisateur a un vrai shell dans l'onglet Editor, ancre sur le dossier de
travail ; il peut le relancer ou le tuer, et le terminal survit a un changement
d'onglet.

## A lire avant d'ecrire

- `web/src/composables/useTerminal.js` (statuts `idle`/`connecting`/`open`/
  `closed`/`error`, frames `open`/`output`/`exit`/`error`, **pas de
  reconnexion automatique**, code de fermeture 4403 si l'`Origin` est refuse)
- `web/src/components/EditorTerminal.vue` : `section[aria-label="Workspace
  terminal"]`, le point + libelle de statut `role=status`, `aria-label="Start a
  new terminal session"`, `aria-label="Kill the terminal session"`, l'hote xterm
  `.editor-term-host`
- `src/api/routes.py` : `WS /api/terminal` (garde d'`Origin`, plateforme POSIX,
  plafond de sessions concurrentes) et `GET /api/terminal` (la liste des sessions
  vivantes, pour le panneau de reglages)

## Tests

1. Ouverture : onglet Editor -> `section[aria-label="Workspace terminal"]`
   present ; le statut devient `open` (lire le libelle `role=status`), le point
   porte la classe "open", et `GET /api/terminal` (fetch) liste exactement une
   session dont le `cwd` est le dossier de travail.
2. Echange clavier : ecrire `echo bi-<marqueur>` + Entree dans le terminal (le
   cibler par sa zone xterm, taper au clavier reel) -> le texte rendu par xterm
   (la ligne `.editor-term-host`) contient le marqueur. C'est la preuve
   aller/retour du PTY, sans dependre du contenu de l'invite.
3. Survie au changement d'onglet : aller sur Benchmark puis revenir sur Editor ->
   la meme session est toujours `open` (`GET /api/terminal` renvoie le meme
   `pid`), aucune nouvelle session n'a ete creee (l'unique socket est un
   singleton).
4. Kill : `aria-label="Kill the terminal session"` -> le statut quitte `open`, le
   bouton New devient actif, et `GET /api/terminal` ne liste plus la session.
5. Nouvelle session : `aria-label="Start a new terminal session"` -> le statut
   repasse a `open` avec un `pid` different.

La garde d'`Origin` (fermeture 4403) n'est **pas** testable depuis le navigateur :
une page Playwright envoie toujours sa propre origine. Ce contrat est couvert par
`tests/test_terminal.py` cote serveur ; mentionner ce delegage en commentaire de
la spec plutot que d'ecrire un faux test.

## Pieges

- Tuer la session en fin de test (le serveur ne doit pas laisser de PTY orphelin).
- Le statut s'attend via le texte du composant, pas via un delai.
- Une seule session concurrente attendue : ne pas ouvrir deux terminaux.

## Acceptation

`npx playwright test e2e/21-editor-terminal.spec.js` au vert, sans shell laisse
vivant (`pgrep -f 'bash'` avant/apres, si la spec en ouvre un).
