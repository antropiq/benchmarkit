# test2e2-13 - WorkBench : piece jointe et capacite vision

Niveau : serveur de fixture pour la branche vision ; le serveur shared couvre la
branche "non supporte" (`.env.test` laisse `SUPPORTS_VISION` commente).
Spec : `web/e2e/13-chat-images.spec.js`.

## Cas d'usage couvert

L'utilisateur attache une image a son message, la voit, peut l'enlever ; et si le
modele ne voit pas d'images, le bouton est fermement ferme.

## A lire avant d'ecrire

- `useChat.js` : `attachImage` (types `IMAGE_MIME_TYPES`, plafond
  `MAX_IMAGE_BYTES`), `clearImage`, `state.pendingImage`
- `web/src/components/WorkBenchView.vue` : le bloc
  `aria-label="Attached image (not sent yet)"`, `aria-label="Remove attached
  image"`, `aria-label="View attached image at full size"`
- `web/src/components/ImageLightbox.vue` (dialog `aria-label="Image preview"`,
  fermeture `aria-label="Close preview"`)
- `WorkBenchInput.vue` : `supportsVision` vient de `GET /api/config`

## Tests (serveur shared, donc vision OFF)

1. Bouton verrouille : le bouton d'attachette est `disabled` et son `title`
   explique que le modele ne gere pas l'entree image ; `GET /api/config` confirme
   `supportsVision: false`.

## Tests (serveur de fixture avec `SUPPORTS_VISION=1`)

2. Attachement : `setInputFiles` sur l'input fichier d'image (PNG 1x1 genere par
   la spec) -> le bloc "Attached image (not sent yet)" apparait avec le nom et la
   taille ; le bouton devient `aria-label="Replace attached image"`.
3. Retrait : `aria-label="Remove attached image"` -> le bloc disparait, le bouton
   `Send message` redevient `disabled` (plus rien a envoyer).
4. Lightbox : `aria-label="View attached image at full size"` ouvre le dialog
   d'apercu, `aria-label="Close preview"` le ferme.
5. Refus d'un type non admis : `setInputFiles` d'un `.txt` -> `.error-line`
   mentionne le type non supporte, aucune piece jointe n'est retenue.

## Pieges

- Deux serveurs dans la meme spec : `test.describe` separes, URLs absolues pour le
  serveur de fixture (voir big picture).
- Ne pas envoyer le message avec l'image si cela consomme une generation : les
  tests s'arretent a la piece jointe prets a partir.

## Acceptation

`npx playwright test e2e/13-chat-images.spec.js` au vert.
