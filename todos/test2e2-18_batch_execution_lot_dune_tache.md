# test2e2-18 - Batch : execution d'un lot, verrouillage, barre, Stop

Niveau : stream reel (les tours du lot sont de vraies generations ; lot a 1
tache, donc 1 generation par essai).
Spec : `web/e2e/18-batch-run.spec.js`.

## Cas d'usage couvert

L'utilisateur laisse l'app enchainer les taches d'un lot, une par tour de chat,
avec une barre de progression ; les seuls gestes possibles sont Stop et Restart.

## A lire avant d'ecrire

- `useBatch.js` : `requestLaunch()` (revalide, passe a l'onglet chat, puis
  `launch()`), `TASK_INSTRUCTION_TEMPLATE`, `state.active` (le drapeau sur lequel
  toute l'UI se verrouille), `outcomes`, `index` (ne compte que les tours
  termines), `stopBatch()`
- `web/src/components/BatchProgressBar.vue` (`section[aria-label="Batch
  progress"]`, `progress` avec `aria-label`)
- `WorkBenchInput.vue` (pendant un batch : textarea `disabled` avec sa
  placeholder, `Send` absent, `Stop` avec `title="Stop the batch"`, Pause absent)
- `TabNav.vue` / `AppBar.vue` (tout est `disabled` pendant un lot)
- `AppBar.vue` : la pill lit `Batch` pendant le lot, puis `batch <outcome>`

## Fixture

Lot temporaire `todos/e2e-lot-<random>/` avec `zz-00-a.md` (big picture court) et
`zz-01-a.md` (une tache dont l'enonce tient en une ligne). Nettoyage complet en
`test.afterAll`.

Les hubs sont desactives par le runner : aucun outil n'est connecte, mais
`ensureToolsAvailable()` juge la **liste declaree** (snapshot de `GET /api/mcp`),
pas les processus. Le lancement est donc autorise si `externals.json` de la
machine declare `fedit` et `console` avec `enable: true` (cas de ce depot) ; le
tour tourne sans outils, ce qui suffit pour tester la boucle.

## Tests

0. Refus (deterministe, sans generation) : avec `page.route('/api/mcp')` renvoyant
   des connecteurs sans `fedit`/`console`, `Launch` echoue et la ligne d'erreur de
   WorkBench (`.error-line`) nomme les connecteurs manquants. Rien n'est lance.
1. Lancement et verrouillage global : ouvrir le dialog, prefix `zz`, dossier du
   lot, `Launch` -> l'onglet actif devient WorkBench, le dialog est ferme, le
   textarea est `disabled`, les trois onglets sont `disabled`, les boutons Batch
   et Settings sont `disabled`, la pill `.status-pill` lit `Batch`.
2. Le tour est bien l'instruction du lot : la bulle utilisateur contient le
   chemin `todos/e2e-lot-<random>/zz-01-a.md` (l'app envoie le chemin, jamais le
   contenu) - verifier par `toContain` sur le texte de la bulle.
3. Barre de progression : `section[aria-label="Batch progress"]` present avec
   `1` comme max ; a la fin du lot, sa valeur vaut 1 et l'etat du lot est
   `finished` (la pill affiche `batch finished` pendant ~2 s : lire l'attribut
   `class` de la pill, pas seulement son texte).
4. Ecran remis a zero entre taches : avec un lot a 2 taches seulement si le cout
   est accepte - sinon, sauter ce cas et le justifier en commentaire.

## Pieges

- Un lot = un tour = une generation. Ne pas multiplier les lots.
- Apres le lot, remettre l'app a l'etat propre (`Restart conversation`) pour ne
  pas laisser l'onglet verrouille pour les specs suivantes.
- La barre ne compte que les tours termines : un Stop laisse `index` en arriere.

## Acceptation

`npx playwright test e2e/18-batch-run.spec.js` au vert, sans lot temporire dans
`todos/`.
