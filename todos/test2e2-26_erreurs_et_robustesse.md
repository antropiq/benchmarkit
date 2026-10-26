# test2e2-26 - Erreurs affichees : ce que l'app dit quand ca casse

Niveau : etat d'UI avec routes interceptees (aucune generation : les pannes sont
simulees, et le test porte sur le message, pas sur la panne reelle).
Spec : `web/e2e/26-error-surfacing.spec.js`.

## Cas d'usage couvert

Le serveur de modele est mort, l'API est injoignable, ou une reponse echoue en
cours de stream : l'utilisateur doit voir **ou** et **pourquoi**, sans ecran
bloque et sans bouton perdu.

## A lire avant d'ecrire

- `useBenchmark.js` : `configError`, `startError` (message du `detail` ou texte
  "Could not reach the API"), l'entree `{type:'error'}` poussee par l'evenement
  `run_error` -> `.error-line` dans `StreamPanel.vue`
- `useChat.js` : `state.error` (`.error-line` de `WorkBenchView.vue`) et le cas
  ou la carte assistant vide est retiree si la requete echoue avant tout contenu
- `useWorkspaceTree.js` / `useEditorFiles.js` : `state.error` de l'arbre,
  `state.message` du panneau (raison + chemin)
- Le contrat d'erreur serveur : `HTTPException` -> `{detail: ...}` (chaine ou
  liste de champs) ; `describeError` cote client sait formater les deux

## Tests

1. Config illisible : `page.route('/api/config')` renvoie 500 `{detail: "Could
   not load questions"}` -> l'alerte `role=alert` de la colonne de stats affiche
   ce message, et le bouton play est `disabled` (pas de demarrage sur une config
   cassée).
2. API injoignable au demarrage : `page.route('**/api/config', route =>
   route.abort())` -> l'alerte contient "Could not reach the API" ; l'app reste
   utilisable (l'en-tete est rendu).
3. Refus de demarrage : intercepter `POST /api/run` (predicat de route filtre sur
   la methode) pour repondre 400 `{detail: "No questions loaded"}` -> l'alerte
   affiche cette raison, la pill ne devient pas `running`, le bouton play est de
   nouveau actif.
4. Erreur en cours de stream de chat : lancer un envoi et faire echouer la
   reponse `/api/chat` (reponse 500 ou corps SSE avec la frame `error`) ->
   `.error-line` est visible sous la conversation, le champ de saisie est de
   nouveau actif, et aucune carte assistant vide ne trainer. (Si la frame `error`
   est simulee, le dire en commentaire.)
5. Erreur de liste pour l'editeur : `page.route('**/api/fs/tree')` renvoie 422 ->
   la ligne d'erreur de l'arbre (`role=alert`) reprend le `detail` du serveur,
   l'onglet Editor reste ouvrable.

## Pieges

- Toujours liberer les routes (`unroute`) ou les limiter a un test : une route
  stubbee qui fuite casse les tests suivants.
- Ne pas confondre les deux lignes d'erreur : le benchmark utilise la colonne de
  stats (`role=alert`) et le flux (`.error-line`), le chat utilise la sienne.
- Le detail en liste de validateurs (422 Pydantic) doit etre rendu lisible : un
  test le verifie avec un `detail` en tableau.

## Acceptation

`npx playwright test e2e/26-error-surfacing.spec.js` au vert.
