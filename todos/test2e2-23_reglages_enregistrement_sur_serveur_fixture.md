# test2e2-23 - Reglages : enregistrement des champs LLM et echantillonnage

Niveau : serveur de fixture obligatoire. **La plus delicate des specs** : lire
 toute la fiche avant d'ecrire.
Spec : `web/e2e/23-settings-save.spec.js`.

## Cas d'usage couvert

L'utilisateur change son endpoint ou son echantillonnage dans la boite de
reglages ; la valeur est persistee et la suite du parcours l'utilise, sans
redemarrer l'app.

## Le piege a connaitre avant tout

`config.ENV_PATH` est ancre sur la **racine du depot** : `POST /api/settings`
reecrit `<racine>/.env`, y compris pour un serveur de fixture. Aucune
configuration parallele n'existe. Donc, obligatoirement :

1. en `test.beforeAll` : lire les octets de `<racine>/.env` et les garder ;
   le fichier doit exister, sinon la spec echoue immediatement (ne rien creer) ;
2. en `test.afterAll` : reecrire ces octets **a l'identique**, puis verifier
   par `git status` (commande locale) que `.env` n'est plus modifie ;
3. un seul serveur de fixture, tue a la fin ; la valeur enregistree pendant le
   test ne survit a rien d'autre.

## A lire avant d'ecrire

- `web/src/components/SettingsDialog.vue` (sections `llm`, `sampling`,
  `verifier` ; le bouton d'enregistrement et son etat pendant la sauvegarde)
- `useBenchmark.js` : `saveSettings` (relit `GET /settings` apres le `POST`)
- `src/api/routes.py` : `POST /settings` (409 pendant un run ; 422 + `detail`
  structure si la config est invalide)
- `src/config.py` : `save_settings` / `_render_env` - le fichier est **regenere
  depuis un gabarit fixe** : ordre de cles fixe, cles optionnelles vides ecrites
  en ligne commentee (`#TOP_K=`)
- Recette du serveur de fixture : big picture, section "Serveur de fixture"

## Tests

1. Aller-retour d'un champ : ouvrir Settings, section LLM, saisir un `endpoint`
   de test (une chaine clairee, ex. `e2e-model-saved`), enregistrer -> le `POST`
   repond 200 avec la nouvelle valeur ; apres rechargement, `GET /api/config`
   (fetch) liste ce modele dans `models` et la barre d'app l'affiche.
2. Persistance : relire `<racine>/.env` -> la ligne `ENDPOINT=` porte la valeur
   saisie, les cles optionnelles restees vides sont des lignes commentees.
3. Echantillonnage : enregistrer `temp` = `0.2` -> la legende du panneau de run
   (`<n> questions · <sampling>`, issue de `GET /api/config` `samplingSummary`)
   contient `temp=0.2` apres rechargement.
4. Rejet : vider `serverUrl` et enregistrer -> reponse 422, le message d'erreur
   de la boite de dialogue est non vide, et l'etat affiche (barre d'app,
   `GET /api/settings`) a garde la valeur precedente.
5. Contrat du verrou : `POST /api/settings` par fetch pendant un run de la
   fixture -> 409 (demarrer le run puis l'annuler tout de suite).

## Pieges

- Attendre la reponse du `POST` avant de recharger la page : sinon la relecture
  de `GET /api/config` est une course.
- Ne pas lancer de benchmark complet ici : un seul run annule suffit.
- Ne jamais toucher `.env.test` : c'est lui qui definit l'endpoint de la suite.

## Acceptation

`npx playwright test e2e/23-settings-save.spec.js` au vert **et** `git status`
ne montre aucune modification de `.env` a la fin.
