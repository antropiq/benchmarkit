# test2e2-08 - Theme, son de fin de session, titre de l'onglet

Niveau : etat d'UI + un seul stream de chat court.
Spec : `web/e2e/08-theme-sound-title.spec.js`.

## Cas d'usage couvert

L'utilisateur travaille l'onglet en arriere-plan : le titre de l'onglet lui dit
si une inference tourne, un son confirme la fin, et le theme est persistant.

## A lire avant d'ecrire

- `web/src/composables/useTheme.js` (cle localStorage, `data-theme`/`color-scheme`)
- `web/src/composables/useTabTitle.js` + `web/src/lib/inference-state.js`
  (`⚡` pendant une inference, `✨` sinon ; titre de base repris de `index.html`)
- `web/src/composables/useSessionSound.js` (cle `benchmarkit:sound`, un seul
  declenchement sur le bord `inferenceRunning` vrai -> faux)
- `web/src/components/ThemeToggle.vue`, `SoundToggle.vue`

## Tests

1. Bascule de theme : le clic sur le bouton `aria-label` "Switch to …" change
   l'attribut porte par `useTheme` et la cle localStorage correspond ; apres un
   `page.reload()`, le theme choisi est restaure. Lire l'attribut par
   `page.getAttribute` sur l'element concerne (justifier l'`evaluate`).
2. Coupe du son : `aria-label="Mute the session-ended sound"` -> `Unmute …`, la
   cle `benchmarkit:sound` vaut `0`, et le choix survit au reload.
3. Titre de l'onglet au repos : `document.title` commence par `✨` et contient le
   titre de base de `index.html` (le lire, ne pas le dupliquer en dur).
4. Titre pendant une inference : envoyer un message court dans WorkBench ; pendant
   que la pill `.status-pill` lit `thinking`, le titre commence par `⚡` ; quand
   la reponse est finie, il repasse a `✨`. Ces deux bords sont l'assertion ; ne
   pas tester le son lui-meme (l'audio n'est pas verifiable headless) - dire en
   commentaire que les deux partagent le meme predicate.

## Pieges

- Les cles localStorage sont partagees : remettre le theme et le son par defaut a
  la fin de chaque test (`test.afterEach`).
- Le bord `⚡` -> `✨` est unique : ne pas envoyer deux messages dans le meme test.

## Acceptation

`npx playwright test e2e/08-theme-sound-title.spec.js` au vert.
