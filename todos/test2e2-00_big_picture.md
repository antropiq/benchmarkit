# E2E du front - big picture (a lire avant toute tache)

But : couvrir les cas d'usage de l'application avec Playwright. Une tache = un
seul fichier de spec, court et autonome. Le decoupage existe pour limiter les
tokens : une tache ecrit peu, et ne relit que les fichiers qu'elle cite.

## Cadre deja en place (ne rien reecrire)

- Runner : `cd web && npm run test:e2e`. Config : `web/playwright.config.js`.
- `testDir: './e2e'` -> une spec = `web/e2e/<nn>-<sujet>.spec.js` (mêmes deux
  chiffres que la tache).
- Serveur teste : la vraie app FastAPI (`uvicorn src.api.server:app`) sur le port
  dedie 8123 (jamais 8000), SPA servie depuis `web/dist` (build deja fait par
  `web/e2e/global-setup.js`). `GET /api/health` -> `{status, env, questions}`.
- Endpoint LLM : `.env.test` exclusivement, jamais `.env`. La config Playwright
  parse `.env.test` (racine du repo) et l'injecte dans l'environnement du serveur
  sous test : elle ecrase toute valeur ambiante (dont le `.env` du
  developpeur). `E2E_ENV_FILE=<fichier>` pour en changer. Si le fichier manque
  le runner echoue fort : c'est voulu, ne jamais le contourner.
- Contenu de `.env.test` (ne pas le supposer ailleurs) : serveur de test dedie,
  `REASONING=off`, `SUPPORTS_VISION` commente donc vision desactivee,
  `QUESTION_SET=teste2e` -> `resources/teste2e.json` = 2 questions (un run
  complet = 2 questions), verificateur pointe sur le meme serveur (un run
  complet est reellement note).
- Le serveur sous test porte `BENCHMARKIT_DISABLE_MCP=1` et
  `BENCHMARKIT_DISABLE_LSP=1` : aucun connecteur MCP ni serveur de langage
  n'est demarre. Une spec qui en veut un demarre son propre serveur (voir
  "serveur de fixture"). Ne jamais retirer ces flags globalement.
- Timeouts : 120 s par test, 30 s par attente - on attend un stream, pas un DOM.
- `workers: 1`, `fullyParallel: false` : specs sequentielles, un seul run de
  benchmark a la fois cote serveur. Une spec ne depend pas de la precedente.

## Deux niveaux de spec (a declarer en en-tete de fichier)

1. stream reel - parle au vrai endpoint de `.env.test`. Interdit de stubber
   `/api/chat`, `/api/run`, `/api/run/events`, `/api/run/stop`. Les reponses
   varient : assertir sur la forme (transitions d'etat, nombre et ordre des
   chunks, libelles produits par l'app, presence d'un badge), jamais sur le
   texte du modele ni sur un nombre exact de tokens.
2. etat d'UI - le test porte sur un composant, pas sur une generation : il peut
   intercepter les routes JSON (`page.route('/api/config')`, `/api/mcp`,
   `/api/lsp`, `/api/fs/*`) pour fabriquer un etat precis (verificateur absent,
   API injoignable, liste de connecteurs). Ne jamais stubber ce que le test
   pretend mesurer, et le dire en commentaire.

## Serveur de fixture (quand une spec change la configuration)

`POST /api/settings` regenere le `.env` du depot : aucune spec ne poste de
reglages sur le serveur partage. Besoin d'une autre config (autre `QUESTION_SET`,
`SUPPORTS_VISION=1`) -> serveur dedie :

1. `spawn('python', ['-m','uvicorn','src.api.server:app','--port','<port>'])`,
   `cwd` = racine du repo, `env` = les couples `KEY=VALUE` lus dans `.env.test`
   (plus la cle qui change, plus `BENCHMARKIT_DISABLE_MCP=1` et
   `BENCHMARKIT_DISABLE_LSP=1`) ;
2. attendre `http://127.0.0.1:<port>/api/health` ;
3. `page.goto('http://127.0.0.1:<port>/')` (ne pas utiliser `baseURL`) ;
4. SIGTERM sur le groupe du process dans `test.afterAll`.

Attention : `config.ENV_PATH` est ancre sur la **racine du depot**, pas sur le
dossier de lancement. Une spec qui poste donc un reglage, un jeu de questions ou
une pastille de memoire ecrit dans `./.env`, `./resources/...` - meme depuis un
serveur de fixture. La discipline est la meme pour ces specs :

1. `test.beforeAll` : lire et garder les octets du fichier cible ;
2. n'ecrire qu'une seule valeur, clairement identifiee ;
3. `test.afterAll` : reecrire les octets d'origine a l'identique ;
4. le critere d'acceptation exige `git status` propre sur le fichier en question.

## Regles d'ecriture (toutes les taches)

1. Une spec = une `test.describe`, 2 a 5 `test()`. Pas de helper mutualise entre
   specs : le duplicata coute moins cher qu'un fichier commun.
2. Selecteurs : role + `aria-label` d'abord (l'app est accessible, il n'y a pas
   de `data-testid`), classes stables ensuite (tableau du fichier de tache).
3. Jamais de `waitForTimeout` : attendre un etat que les composables exposent
   dans le DOM, ou `expect(...).toPass()`.
4. Etat initial propre : bouton `aria-label="Reset to initial state"` pour le
   run, `aria-label="Restart conversation"` pour le chat.
5. Ne rien committer et ne pas modifier les fichiers de l'app, sauf la fixture
   explicitement demandee par une tache, toujours nettoyes a la fin. Le seul
   fichier a creer est la spec demandee ; le stage/commit puis le deplacement du
   fichier de tache vers `todos/done/` sont le contrat du mode batch, pas a
   contester ici.
6. JS simple, imports limites a `@playwright/test` et aux modules `node:*`.
   `page.evaluate` seulement pour lire un etat non expose au DOM, justifie en
   commentaire.
7. Echec lisible : message d'attente explicite, pas de `try/catch` qui avale.

## Budget de generations (le lot entier est facture)

Chaque tour de stream est un appel reel au serveur de test. Regles :

- une spec = **au plus 2 generations** (run complet = 2 questions, compte comme
  une seule serie) ;
- un run complet n'est lance que par les specs qui l'affirment (02, 04) ; les
  autres annulent tot (`aria-label="Stop run"`) ;
- preferer l'apercu du brouillon, les etats stubbes et les contrats HTTP quand
  cela suffit a prouver le cas d'usage.

## Index du lot

| # | Sujet | Spec |
|---|---|---|
| 01 | Demarrage : config affichee, onglets, etat vide | `01-app-boot` |
| 02 | Run : stream, progression, mesures | `02-run-stream` |
| 03 | Run : annulation et remise a zero | `03-run-cancel` |
| 04 | Rechargement : rejeu du run en cours | `04-run-reconnect` |
| 05 | Exclusivite run / chat / batch / reglages | `05-mutual-exclusion` |
| 06 | Reglages en lecture seule | `06-settings-readonly` |
| 07 | Jeux de questions | `07-question-sets` |
| 08 | Theme, son, titre d'onglet | `08-theme-sound-title` |
| 09 a 16 | WorkBench : stream, Stop/Pause, compositeur, session, images, mesures, rendu, panneaux MCP/LSP | `09-…` a `16-…` |
| 17, 18 | Batch : dialog + listing, puis execution | `17-batch-dialog`, `18-batch-run` |
| 19 a 22 | Editeur : arbre, fichier, terminal, diff + mode | `19-…` a `22-…` |
| 23, 24 | Reglages : enregistrement, sections MCP/LSP | `23-settings-save`, `24-settings-hubs` |
| 25 | Dossier de travail (changement + refus) | `25-working-location` |
| 26 | Erreurs affichees et robustesse | `26-error-surfacing` |
| 27 | Separateurs de panneaux | `27-splitters-accessibility` |
| 28, 29 | Pastilles de memoire ; resultats telechargeables | `28-memory-bullets`, `29-results-download` |

Le lot se lance tel quel en mode batch : prefix `test2e2-`, dossier `todos`,
big picture inclus. `todos/test2e2-00_big_picture.md` est deja le fichier `00` ;
les taches vont de `01` a `29` (verifie par `GET /api/batch/tasks`).

## Critere d'acceptation (identique pour toutes les taches)

`cd web && npx playwright test e2e/<nn>-<sujet>.spec.js` passe au vert, sans
`test.skip` ni `test.only`. Si le serveur de test est injoignable, la tache
n'est pas finie : le signaler plutot que d'ajouter un skip.
