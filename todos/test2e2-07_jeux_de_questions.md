# test2e2-07 - Jeux de questions : decouverte et selection

Niveau : serveur de fixture obligatoire (la selection ecrit dans un `.env`).
Spec : `web/e2e/07-question-sets.spec.js`.

## Cas d'usage couvert

L'utilisateur depose un JSON dans `resources/` et le choisit dans l'UI : le
nombre de questions du run change immediatement.

## A lire avant d'ecrire

- `web/src/components/SettingsDialog.vue` (section `question-set`),
  `useBenchmark.js` : `loadQuestionSets`, `saveQuestionSet`
- `src/config.py` : `list_question_sets()` (tout `*.json` de `resources/` avec une
  cle `questions`), `questions_path()`, `QUESTION_SET_DEFAULT`
- `src/api/routes.py` : `GET /question-sets`, `POST /question-sets` (409 pendant
  un run, 422 pour un id inconnu)
- Recette du serveur de fixture : section "Serveur de fixture" du big picture.

## Fixture de la spec

Serveur dedie demarre avec l'environnement lu dans `.env.test`. Un jeu temporaire `resources/teste2e-tmp-<random>.json`
(2 questions, meme forme que `resources/teste2e.json`) est cree dans
`test.beforeAll` et **supprime** dans `test.afterAll`. Ne jamais toucher aux jeux
deja commits. La selection d'un jeu ecrit `QUESTION_SET` dans le `.env` de la
**racine du depot** : sauvegarder ses octets en `beforeAll`, les restaurer en
`afterAll`
(discipline "serveur de fixture" du big picture).

## Tests

1. Decouverte : `GET /api/question-sets` liste un `id` par fichier de
   `resources/` (avec `count`), y compris le jeu temporaire ; la section
   "Question set" du dialog les affiche et selectionne celui de `.env.test`.
2. Selection : choisir le jeu temporaire puis valider -> le `POST` a reussi,
   `GET /api/question-sets` renvoie ce `current`, et `GET /api/config` affiche le
   `questionCount` correspondant. Le `.env` du serveur de fixture (le fichier
   temporaire) contient la nouvelle valeur : le lire.
3. Rejet : `POST /api/question-sets` avec un id inexistant par fetch -> 422 avec
   un `detail` qui nomme le jeu ; l'app n'a pas change d'etat.

## Pieges

- Ce serveur de fixture n'est pas celui du port 8123 : toutes les URLs sont
  absolues. Ne jamais poster de jeu de questions sur le serveur partage.
- Attendre la fin du `POST` (reponse) plutot qu'un delai avant de relire `config`.

## Acceptation

`npx playwright test e2e/07-question-sets.spec.js` au vert, sans fichier
`teste2e-tmp-*` laisse dans `resources/`.
