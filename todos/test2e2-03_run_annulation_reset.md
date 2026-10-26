# test2e2-03 - Annulation d'un run, puis remise a zero

Niveau : stream reel.
Spec : `web/e2e/03-run-cancel.spec.js`.

## Cas d'usage couvert

L'utilisateur annule un run en cours, et peut immédiatement en relancer un propre.

## A lire avant d'ecrire

- `useBenchmark.js` : `stop()` (409 gere), l'evenement `run_finished` avec
  `cancelled`, `resetToIdle()`, l'entree `{type:'cancelled'}`
- `src/api/routes.py` : `POST /run/stop` (409 si aucun run), `GET /run`
- `StreamPanel.vue` : `.result-banner.warning` ("Benchmark cancelled")

## Tests

1. Annulation en cours de stream : demarrer, attendre la premiere carte
   `.question-card`, cliquer `aria-label="Stop run"` (le bouton passe par
   `Stopping…`). Attendre `.result-banner.warning` puis verifier : la pill
   d'etat est `cancelled`, le compteur est revenu a sa valeur initiale, le bouton
   play est de nouveau actif. La banniere "cancelled" disparait d'elle-meme au
   bout de quelques secondes : ne pas l'assertionner apres ce delai.
2. Annulation entre questions : accepter le fait qu'on ne maitrise pas le moment
   (pas de sleep). Verifier a la place l'invariant : apres l'annulation,
   `GET /api/run` (fetch) ne renvoie plus `status: "running"`.
3. Reset apres fin : avec un run termine (ou annule), cliquer
   `aria-label="Reset to initial state"` -> plus aucune `.question-card`,
   `.chronometer` a `00:00:00`, `Score 0/0`, `progress` a 0. Cet etat-la est
   verifiable sans nouvelle generation.
4. Refus d'annulation hors run : `POST /api/run/stop` par fetch doit repondre 409
   quand aucun run ne tourne (verifie le contrat derriere le bouton desactive).

## Pieges

- Ne pas confondre le bouton : pendant le run c'est "Stop run", hors run c'est
  "Reset to initial state" (meme bouton, `aria-label` different).
- Ne pas enchainer un deuxieme run complet : si le test 1 a besoin de relancer,
  annuler a nouveau immediatement.

## Acceptation

`npx playwright test e2e/03-run-cancel.spec.js` au vert.
