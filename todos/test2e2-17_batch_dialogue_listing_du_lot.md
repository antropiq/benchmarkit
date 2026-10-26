# test2e2-17 - Batch : dialog, listing du lot, refus avant lancement

Niveau : etat d'UI + contrat HTTP (aucune generation : rien n'est lance).
Spec : `web/e2e/17-batch-dialog.spec.js`.

## Cas d'usage couvert

L'utilisateur prepare un lot de taches : il choisit le prefix et le dossier,
l'app lui dit combien de taches elle a trouvees, lesquelles elle ecarte et
pourquoi, avant toute execution.

## A lire avant d'ecrire

- `web/src/components/BatchModeDialog.vue` (champs prefix/dossier, la checkbox
  "Include the big picture", les lignes de compte/skip/lot, `Launch`)
- `web/src/composables/useBatch.js` : `PREFIX_PATTERN`, `validate()`, `canLaunch`
- `src/core/batch.py` : `task_lot()` (le `NN==0` = big picture, `taskCount` = le
  plus grand numero, `skipped` avec sa raison, `truncated`), `resolve_lot_dir()`
  (les trois verrous), `GET /api/batch/tasks?prefix=&dir=`

## Fixture de la spec

Creer un lot temporaire `todos/e2e-lot-<random>/` dans le depot : `xx-00-a.md`
(big picture), `xx-01-a.md`, `xx-02-b.md`, `xx-1x-bad.md` (numero non separe),
`xx-nope.md` (pas de numero). `test.afterAll` supprime le dossier. Ne rien
deplacer vers `todos/done/`.

## Tests

1. Ouverture : `aria-label="Batch mode"` -> dialog `aria-labelledby="batch-dlg-title"`
   visible, avec ses deux champs et `Launch` `disabled` (prefix vide).
2. Listing : taper `xx` dans le prefix, quitter le champ (le listing part sur le
   `blur`) -> la ligne `role=status` annonce 2 taches (`taskCount` = 2) ; les deux
   fichiers mal nommes apparaissent dans les lignes "skipped" avec leur raison ;
   le big picture est mentionne et la checkbox "Include the big picture" est
   cochee ; `Launch` devient actif.
3. Prefix invalide : `xx*` -> message d'erreur du compositeur (motif
   "must start with a letter or digit"), aucun appel a `/api/batch/tasks`
   (verification par ecoute de requete).
4. Dossier inexistant : `dir` = `nope-<random>` -> la ligne d'erreur cite le
   dossier (le serveur repond 404) et `Launch` reste `disabled`.
5. Verrous du chemin : `dir` = `../etc` et `dir` = `/etc` par fetch direct sur
   `/api/batch/tasks` -> 422 avec un `detail` qui nomme la regle. Ne pas passer
   par l'UI pour ce test (le contrat est ce qui compte).

## Pieges

- Ne jamais cliquer `Launch` dans cette spec : c'est la tache 18 qui execute.
- Un prefix qui ne trouve rien n'est pas une erreur : verifier le compte a zero
  et le `title` du bouton ("No task matches this prefix").

## Acceptation

`npx playwright test e2e/17-batch-dialog.spec.js` au vert, sans dossier
`e2e-lot-*` laisse dans `todos/`.
