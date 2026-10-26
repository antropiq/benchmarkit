# test2e2-28 - Pastilles de memoire : configurer un message type et s'en servir

Niveau : etat d'UI + ecriture du fichier des creneaux (a restaurer).
Spec : `web/e2e/28-memory-bullets.spec.js`.

## Cas d'usage couvert

L'utilisateur enregistre des messages types (prompts reutilisables) sur 9
pastilles, les colore, puis les injecte dans le brouillon d'un clic.

## Le piege a connaitre avant tout

`POST /api/memory` ecrit `resources/memory_registration.json` (fichier de la
machine, gitignore, **auto-repare** par le serveur). Comme pour les reglages :

1. `test.beforeAll` : lire et garder les octets du fichier (s'il manque, le
   serveur le recreera - le notifier, ne pas echouer) ;
2. `test.afterAll` : reecrire ces octets a l'identique ;
3. n'utiliser que le creneau `index: 0` pour les ecritures.

## A lire avant d'ecrire

- `web/src/components/MemoryBullets.vue` (les 9 pastilles, `role=group`
  `aria-label="Memory bullets"`, le clic = insertion, le clic long / bouton de
  configuration qui ouvre le dialog)
- `web/src/components/RegistrationMemoryDlg.vue` (dialog
  `aria-labelledby="memory-dlg-title"`, champ Message, champ couleur, `Cancel`,
  `Save`/`Saving…`, la ligne d'erreur `role=alert`)
- `web/src/composables/useMemory.js` : `loadMemory`, `saveMemory` (plafond de
  caracteres cote client), les constantes qui **doivent miroiter** `src/config.py`
  (`MEMORY_SLOT_COUNT` = 9, couleur hex a 6 chiffres, contenu <= 32000)
- `useChat.js` : `appendDraft` (ajoute, n'ecrase jamais)

## Tests

1. Structure : `GET /api/memory` (fetch) renvoie 9 creneaux ; la barre en affiche
   9, avec la couleur par defaut (#808080) quand le contenu est vide.
2. Enregistrement : ouvrir la pastille 1, saisir un texte marque, enregistrer ->
   `POST /api/memory` repond 200 et renvoie les 9 creneaux avec le nouveau texte a
   l'index 0 ; le fichier sur disque contient cette valeur.
3. Injection : un texte est deja dans le brouillon -> clic sur la pastille -> le
   brouillon contient l'ancien texte **puis** le nouveau, separes par une ligne
   vide ; le focus revient dans le textarea.
4. Effacement : enregistrer un contenu vide remet le creneau a vide (et
   n'insere rien dans le brouillon).
5. Refus cote serveur : `POST /api/memory` par fetch avec une couleur invalide
   (`"#ff00"`) ou un index hors plage -> 422 avec un `detail` non vide ; l'etat
   affiche n'a pas change.

## Acceptation

`npx playwright test e2e/28-memory-bullets.spec.js` au vert, et le fichier
`resources/memory_registration.json` identique a son etat de depart.
