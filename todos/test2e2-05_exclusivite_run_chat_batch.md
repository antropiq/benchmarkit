# test2e2-05 - Exclusivite : un run bloque chat, batch et reglages

Niveau : mixte (une generation de run + assertions d'UI + contrats HTTP).
Spec : `web/e2e/05-mutual-exclusion.spec.js`.

## Cas d'usage couvert

Pendant un benchmark, l'utilisateur ne peut ni discuter, ni lancer un lot, ni
changer la configuration : l'app doit le rendre impossible a l'ecran et le refuser
cote serveur.

## A lire avant d'ecrire

- `useBenchmark.js` : `setTab` (refuse `chat` pendant un run), `openSettings`
  (refuse si `running` ou batch), `canStop`
- `web/src/components/TabNav.vue` (boutons desactives), `AppBar.vue` (boutons
  batch/reglages `:disabled`)
- `src/api/routes.py` : `POST /chat` 409 pendant un run ; `POST /run` 409 si une
  reponse chat est en cours ; `POST /settings` et `POST /question-sets` 409
  pendant un run ; `POST /mcp/restart` 409 dans les deux cas

## Tests

1. Onglet WorkBench interdit pendant un run : lancer un run, verifier que l'onglet
   `WorkBench` est `disabled` (et que l'onglet `Editor` ne l'est pas - c'est
   l'invariant ecrit dans `TabNav.vue`).
2. Reglages et batch inaccessibles : les boutons `aria-label="Settings (available
   when idle)"` et `aria-label="Batch mode"` sont `disabled` pendant le run.
3. Chat refuse cote serveur : `POST /api/chat` (fetch, avec un body minimal
   valide) renvoie 409 pendant le run, avec un `detail` non vide.
4. Run refuse sous un chat : depuis l'onglet WorkBench a l'etat idle, lancer une
   reponse puis `POST /api/run` par fetch -> 409. (Ne pas cliquer le bouton play :
   il est desactive ; c'est le contrat HTTP qui est teste.) Penser a arreter la
   reponse chat en cours avec `aria-label="Stop generating"`.
5. Un run qui a fini ne bloque plus : apres la fin du run de 1, l'onglet WorkBench
   est de nouveau cliquable.

## Pieges

- Un seul run complet pour toute la spec : les tests 1, 2, 3, 5 se font pendant /
  autour du meme run. Le test 4 est un `test()` distinct, apres remise a zero.
- Les fetchs ne doivent pas creer d'etat persistent : un `POST /api/chat` 409 ne
  demarre rien (le serveur refuse avant), c'est sans risque.
- `Stop generating` puis attendre `phase` idle avant le test suivant.

## Acceptation

`npx playwright test e2e/05-mutual-exclusion.spec.js` au vert.
