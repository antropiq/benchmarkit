# test2e2-25 - Dossier de travail : le changer, et l'arbre le suit

Niveau : serveur de fixture (le changement redemarre les hubs du process sous
test ; il n'est **jamais** persiste).
Spec : `web/e2e/25-working-location.spec.js`.

## Cas d'usage couvert

L'utilisateur pointe les outils et l'onglet Editor vers un autre dossier du
disque, d'un seul clic ; un dossier interdit est refuse avec sa raison.

## A lire avant d'ecrire

- `web/src/components/SettingsDialog.vue` (section `working-dir` : le champ, le
  bouton `Set`, son libelle "Reloading…" pendant l'appel)
- `useBenchmark.js` : `loadPathLocation`, `setPathLocation` (relit le payload,
  met a jour `pathLocationCurrent`, re-racine l'arbre)
- `src/api/routes.py` : `GET`/`POST /path-location` - 409 pendant un run ou un
  chat, 422 si vide / non absolu / inexistant / sous une entree de
  `lsp_connectors/restricted-paths.txt` ; la valeur est process-local (pas de
  persistance)
- `web/src/components/AppBar.vue` : `.appbar-workspace-path` montre la valeur
  confirmee par le serveur

## Fixture

Serveur dedie demarre avec le contenu de `.env.test` **plus**
`BENCHMARKIT_DISABLE_MCP=1` et `BENCHMARKIT_DISABLE_LSP=1` (les hubs repondent
mais ne connectent rien, et `POST /path-location` a besoin que les deux hubs
existent : c'est le cas). Deux dossiers temporaires via `os.tmpdir()` : une cible
valide (avec un fichier temoin dedans) et un chemin inexistant.

## Tests

1. Valeur affichée : la section "Working directory" presente le chemin confirme
   par `GET /api/path-location`, identique a `.appbar-workspace-path`.
2. Changement : saisir le dossier temporaire valide, cliquer `Set` -> le `POST`
   repond 200 ; `GET /api/path-location` (fetch) renvoie le chemin **canonicalise**
   (resolu), la barre d'app est mise a jour sans rechargement.
3. L'arbre suit : ouvrir l'onglet Editor -> `GET /api/fs/tree` racine sur le
   nouveau dossier, le fichier temoin y apparait, le nom de la racine de l'arbre
   est celui du dossier temporaire.
4. Refus (par fetch, trois variantes) : chemin inexistant, chemin relatif, et un
   chemin sous `/etc` -> 422 dans les trois cas, avec un `detail` qui nomme la
   raison ; `GET /api/path-location` n'a **pas** change.
5. Retour a la valeur initiale : le serveur de fixture est tue a la fin ; verifier
   qu'aucun fichier du depot n'a ete modifie (`git status`).

## Pieges

- Ne pas pointer le serveur **partage** vers un autre dossier : cela
  re-racinerait les hubs de tous les autres specs.
- Le bouton `Set` est `disabled` tant que le champ egale la valeur confirmee :
  saisir puis verifier qu'il s'active (c'est deja un cas d'UI en soi).
- Attendre la reponse du `POST` avant d'ouvrir l'onglet Editor.

## Acceptation

`npx playwright test e2e/25-working-location.spec.js` au vert.
