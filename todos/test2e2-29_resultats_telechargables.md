# test2e2-29 - Resultats : le lien de telechargement et le fichier ecrit

Niveau : stream reel (un seul run complet, 2 questions) + lecture d'un
telechargement.
Spec : `web/e2e/29-results-download.spec.js`.

## Cas d'usage couvert

Un run termine laisse un `results.json` telechargeable depuis l'app, et ce fichier
raconte bien le run qui vient d'avoir lieu.

## A lire avant d'ecrire

- `web/src/App.vue` : le lien `Download results.json` (`href="/api/results"`,
  `download="results.json"`), affiche seulement si `state.hasResults`
- `useBenchmark.js` : `hasResults` vient de `GET /api/config` (`RESULTS_PATH.is_file()`)
- `src/api/routes.py` : `GET /results` (404 si absent, sinon `FileResponse`)
- `src/core/utils.py` : `save_results` (ecrit a la fin du run) et la forme de
  `job.to_output()` ; `src/core/job.py` pour les champs agreges

## Tests

1. Lien present seulement apres un run : au debut de la spec, lire
   `GET /api/config` -> `hasResults` est un booleen. Si faux, le lien n'est pas
   dans le DOM ; si vrai, il l'est. Ne pas ecrire l'attente en dur.
2. Run complet puis telechargement : lancer le run (2 questions), attendre
   `.result-banner.success` ; `GET /api/config` a `hasResults: true` et le lien
   est visible. Capturer l'evenement `download` du clic sur le lien : le nom du
   fichier est `results.json`, le JSON se relit.
3. Cohérence du fichier : le JSON telecharge contient une entree par question du
   jeu actif (nombre = `questionCount` de `GET /api/config`), chaque entree porte
   sa question, sa reponse, un verdict booleen et des chiffres de temps/tokens
   (types verifies, valeurs non verifiees). Les champs exacts se lisent dans
   `RunJob.to_output()` : ne rien inventer.
4. Fraicheur : l'identifiant du run (ou l'horodatage) present dans le fichier
   correspond au run lance dans cette spec - pas a un run precedent. Comparer avec
   le `runId` renvoye par `POST /api/run` ou lu dans `GET /api/run`, si
   `to_output()` en porte un ; sinon verifier que la duree totale est non nulle.

## Pieges

- `results.json` est gitignore : le laisser sur disque, ne pas le supprimer.
- Ne pas relancer un run complet pour le test 4 : un seul run, relu deux fois.

## Acceptation

`npx playwright test e2e/29-results-download.spec.js` au vert.
