# test2e2-09 - WorkBench : envoi, streaming, fin de tour

Niveau : stream reel (le vrai endpoint de `.env.test`).
Spec : `web/e2e/09-chat-stream.spec.js`.

## Cas d'usage couvert

L'utilisateur pose une question dans WorkBench et voit la reponse s'ecrire, avec
les indicateurs d'attente, puis l'app revient a l'etat idle et il peut relancer.

## A lire avant d'ecrire

- `web/src/composables/useChat.js` : `send()`, `phase` (`idle`/`waiting`/
  `reasoning`/`answering`), `applyEvent` (frames `token`, `usage`, `done`,
  `error`), `finish()`
- `web/src/components/WorkBenchView.vue` (`.chat-bubble`, `article.question-card`,
  `.chat-typing`, `.error-line`, `.stream-empty`) et `WorkBenchInput.vue`

## Tests

1. Envoi par Entree : taper dans `textarea[aria-label="WorkBench input"]` puis
   `Enter` -> le message utilisateur apparait dans une `.chat-bubble`, une carte
   assistant est ajoutee, et le brouillon est vide.
2. Indicateur d'attente puis stream : pendant le tour, `.chat-typing`
   ("Thinking…") est visible tant que la carte est vide, puis le contenu de la
   carte croit (mesurer la longueur du texte a deux instants) - preuve du stream.
3. Etat pendant/apres : pendant le tour le bouton `aria-label="Stop generating"`
   est present et `Send message` absent ; a la fin (`done`) c'est l'inverse, et la
   pill de la barre d'app est repassee a `idle`.
4. Tour suivant sur le meme fil : un deuxieme envoi conserve le premier message
   (2 bulles utilisateur) - la conversation est bien re-postee a chaque tour.
5. Envoi refuse pendant un run : deja couvert par 05 ; ici verifier seulement
   l'UI : `canSend` est faux quand le brouillon est vide (bouton `Send message`
   `disabled`).

## Pieges

- Deux tours = deux generations : les tests 1-4 se suivent dans un seul `test()`
  ou partagent une conversation ; ne pas multiplier les allers-retours.
- `aria-label="Restart conversation"` en debut de test pour un etat propre (il
  n'est actif que s'il y a quelque chose a remettre a zero).
- Aucune assertion sur le contenu de la reponse.

## Acceptation

`npx playwright test e2e/09-chat-stream.spec.js` au vert.
