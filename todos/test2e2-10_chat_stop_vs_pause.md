# test2e2-10 - WorkBench : Stop detruit le fragment, Pause le garde

Niveau : stream reel (deux generations volontairement interrompues).
Spec : `web/e2e/10-chat-stop-pause.spec.js`.

## Cas d'usage couvert

Deux gestes distincts pendant une reponse : Stop = "j'abandonne ce tour",
Pause = "je coupe mais je garde ce qui est ecrit, et je commente".

## A lire avant d'ecrire

- `useChat.js` : `stop()` (flag `userStopped`, le fragment est ecarte), `pause()`
  (flag `pausePending`, `POST /api/chat/pause`, frame terminale `pause_ack` qui
  garde le fragment et met `message.paused = true`), `finish()`
- `web/src/components/WorkBenchView.vue` : le caption `.chat-paused`
  ("Interrupted by you") et `.chat-truncated` ("Cut short by the server")
- `src/api/routes.py` : `POST /api/chat/pause` (200, pas de body)

## Tests

1. Stop : envoyer, attendre que du texte apparaisse, cliquer
   `aria-label="Stop generating"`. Attendre le retour a l'etat idle (bouton
   `Send message` revenu). Attendu : la carte assistant de ce tour a disparu
   (fragment ecarte) alors que la bulle utilisateur reste (pour edit-and-resend)
   ; le chronometre de session (`ChatStopwatch`) est revenu a `00:00:00`.
   Assertir sur le nombre de cartes, pas sur le texte.
2. Pause : envoyer, attendre du texte, cliquer `aria-label="Pause — keep the
   partial reply, add a comment"`. Le bouton devient `Pausing…` et `disabled`.
   Attendu : `POST /api/chat/pause` a repondu 200 (ecouter la reponse), le
   fragment reste visible **avec** le caption `.chat-paused`, la zone de saisie
   est de nouveau utilisable.
3. Reprise apres pause : taper un commentaire et l'envoyer -> le tour interrompu
   est toujours present au-dessus (le fragment a ete re-poste au modele, ce que
   prouve sa persistance a l'ecran), un nouveau tour demarre.
4. Pause impossible hors stream : le bouton Pause n'existe pas a l'etat idle
   (`v-if="streaming"`), et `POST /api/chat/pause` par fetch repond 409 quand
   aucune reponse n'est en cours (verifier le contrat).

## Pieges

- Stop et Pause ne s'ecrivent pas pareil a l'ecran : l'un detruit, l'autre garde.
  C'est LA distinction que le test doit demontrer, avec le compteur de cartes.
- Le bouton Pause disparait une fois le tour fini : le cibler pendant le stream.
- Le serveur ne coupe pas au premier octet : attendre que du texte soit visible
  avant d'agir, sinon on teste une course.

## Acceptation

`npx playwright test e2e/10-chat-stop-pause.spec.js` au vert.
