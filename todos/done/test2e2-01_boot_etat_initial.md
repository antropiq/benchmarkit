# test2e2-01 - Demarrage : config affichee, onglets, etat vide

Niveau : etat d'UI (le serveur shared suffit, aucun stream).
Spec : `web/e2e/01-app-boot.spec.js`.

## Cas d'usage couvert

L'utilisateur ouvre l'app : elle doit lui dire contre quel serveur et quel modele
il va travailler, quels onglets existent, et qu'aucun run n'a encore eu lieu.

## A lire avant d'ecrire

- `web/src/App.vue` (structure des trois vues, `v-show`)
- `web/src/components/AppBar.vue`, `TabNav.vue`, `RunControls.vue`, `Chronometer.vue`
- `src/api/routes.py` : `GET /config` (payload `models`, `serverUrl`,
  `questionCount`, `totalTests`, `verifierConfigured`, `runStatus`, `hasResults`)

## Assertions (une par test)

1. Barre d'app : le nom du modele de `.env.test` apparait dans `.brand-sub`, et la
   ligne `on <SERVER_URL>` aussi. L'argument du test doit venir de
   `GET /api/config` (fetch dans la spec), jamais d'une chaine ecrite en dur.
2. Trois onglets `role=tab` : Benchmark selectionne, WorkBench et Editor presents.
3. Colonne de stats : `.chronometer` vaut `00:00:00`, les cartes affichent
   `Score 0/0`, `Question in progress --`, et le meta `2 questions · <sampling>`
   (nombre issu de `questionCount`).
4. Zone de stream vide : titre "Ready to benchmark", bouton play
   `aria-label="Run benchmark"` active, bouton stop `aria-label="Reset to initial
   state"` desactive.
5. Chemin du workspace visible dans `.appbar-workspace-path` (vient de
   `GET /api/path-location`).

## Contre-exemples a tester (1 test)

Avec `page.route('/api/config')` renvoyant `verifierConfigured: false` : le
message d'alerte "No verifier agent configured" est visible. Commenter que la
route est stubbee parce que le test porte sur l'alerte, pas sur la config.

## Acceptation

`npx playwright test e2e/01-app-boot.spec.js` au vert.
