# test2e2-02 - Run de benchmark : stream, progression, mesures

Niveau : stream reel (le vrai endpoint de `.env.test`, 2 questions).
Spec : `web/e2e/02-run-stream.spec.js`.

## Cas d'usage couvert

L'utilisateur lance le benchmark et regarde les reponses s'ecrire question par
question, avec la progression, le score et la vitesse.

## A lire avant d'ecrire

- `web/src/composables/useBenchmark.js` : evenements SSE (`run_started`,
  `model_started`, `question_started`, `token`, `answer_done`, `question_done`,
  `run_finished`) et leur effet sur `state`
- `web/src/components/StreamPanel.vue`, `QuestionCard.vue`, `MetricCard.vue`,
  `App.vue` (colonne stats + `.progress-row`)

## Plan du test principal (run complet = 2 questions, ne pas le relancer)

Un seul `test()` fait le run complet une fois, les assertions suivantes sont des
etapes de ce meme test (le lot ne doit pas generer deux fois).

1. Clic sur `aria-label="Run benchmark"` : le bouton devient
   `aria-label="Benchmark running"`, la pill `.status-pill` lit `running`.
2. `.chronometer` affiche `Starting…` (warmed-up) puis une duree `hh:mm:ss` non
   nulle quand la premiere question demarre.
3. Une premiere carte `.question-card` apparait avec sa legende
   `Question <n> - <secondes>` ; son contenu croit (le nombre de caracteres de
   son `.markdown-stream` augmente entre deux lectures) - c'est la preuve du
   stream, sans dependre du texte.
4. A la fin de la reponse : badge de verdict (`.badge.green`, `.badge.red` ou
   `.badge.violet` "Cut short") et la legende `<duree>s · <n> tokens · <x> tok/s`.
5. Le compteur `.progress-row .mono` passe a `1/2` puis `2/2` ; la valeur de
   `progress[aria-label="Run progress"]` croit pareil.
6. A la fin : `.result-banner.success` ("Benchmark finished"), la pill repasse a
   `finished`, la carte `Score` affiche `<succes>/2`, et le lien
   `Download results.json` apparait.
7. Apres le run : le bouton play est a nouveau actif et
   `aria-label="Reset to initial state"` est disponible.

Chaque spec redemarre de son propre etat : cliquer "Reset to initial state" en
debut de test quand `GET /api/run` n'est pas `idle`, jamais en fin de test.

## Interdictions

Aucune assertion sur le contenu semantique d'une reponse, sur un nombre de tokens
precis, ni sur un temps inferieur a une borne. Attendre les etats, jamais le temps.

## Acceptation

`npx playwright test e2e/02-run-stream.spec.js` au vert.
