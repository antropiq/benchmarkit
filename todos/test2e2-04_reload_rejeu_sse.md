# test2e2-04 - Rechargement de page : le run en cours est retrouve

Niveau : stream reel (une seule generation).
Spec : `web/e2e/04-run-reconnect.spec.js`.

## Cas d'usage couvert

L'utilisateur rafraichit l'onglet pendant un run : rien n'est perdu, l'ecran se
reconstruit et le stream reprend la ou il en etait.

## A lire avant d'ecrire

- `useBenchmark.js` : `loadConfig()` (rejoue si `runStatus` en cours),
  `reconnectViaSnapshot()` (appelle `GET /api/run`, rejoue `events`, puis
  `connect(lastSeq)`), le buffer numerote `since=` de `/api/run/events`
- `src/api/events.py` : `EventBus` (replay par seq)
- `src/api/routes.py` : `GET /run` -> `{status, runId, lastSeq, events}`

## Tests

1. Rechargement en cours de run : demarrer, attendre une premiere carte
   `.question-card` (ou le compteur `1/2` si la premiere question est finie),
   puis `page.reload()`. Apres reload : l'app est de nouveau en etat `running`
   (pill), les cartes deja produites sont rejouees (au moins une `.question-card`
   avec un verdict), et le run va jusqu'a son terme sans relancer `POST /api/run`
   (controler par `page.waitForResponse` que seul un `GET /api/run` a eu lieu
   apres le reload - c'est la preuve qu'il n'y a pas de double generation).
2. Pas de rejeu apres annulation : `GET /api/config` (fetch) renvoie
   `runStatus: "cancelled"` ou `idle`, et apres un reload l'app affiche l'etat
   vide (aucune carte). Decrit l'invariant "un run annule ne se rejoue pas".
3. Continuite du buffer (lecture du contrat, sans UI) : pendant un run, fetch
   `GET /api/run` et verifier que `events` porte des `seq` strictement croissantes
   et sans trou, et que `lastSeq` vaut la derniere - c'est ce sur quoi le rejeu
   `?since=` s'appuie.

## Pieges

- Le reload doit intervenir pendant le run, pas apres : verifier la pill avant de
  recharger, sinon le test 1 ne prouve rien.
- Le bus SSE ne rouvre pas de run : `POST /api/run` ne doit jamais etre rappele.
- Pas de deuxieme run complet : le test 1 mene le run jusqu'a sa fin, les tests 2
  et 3 tournent sur l'etat laisse (ou sur des fetchs seuls).

## Acceptation

`npx playwright test e2e/04-run-reconnect.spec.js` au vert.
