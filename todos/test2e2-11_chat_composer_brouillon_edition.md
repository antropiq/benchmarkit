# test2e2-11 - Compositeur : apercu du brouillon, edit-and-resend, redemarrage

Niveau : etat d'UI majoritairement ; une seule generation reelle (le resend).
Spec : `web/e2e/11-chat-composer.spec.js`.

## Cas d'usage couvert

L'utilisateur prepare son message (apercu markdown), se trompe, recupere le
dernier message pour le corriger, ou repart de zero.

## A lire avant d'ecrire

- `useChat.js` : `setDraft` (apercu debounced 200 ms via `renderMarkdown`),
  `appendDraft`, `editLast()` (recharge le brouillon, supprime ce message et sa
  suite), `reset()`
- `web/src/components/WorkBenchView.vue` : l'apercu
  `aria-label="Message preview (not sent yet)"`, le bouton
  `aria-label="Edit and resend"` (visible seulement sur la derniere bulle
  utilisateur, a l'etat idle)
- `web/src/components/MemoryBullets.vue` + `useMemory.js` (les 9 pastilles)

## Tests

1. Apercu du brouillon : taper une ligne de markdown (une liste ou du gras) dans
   le textarea -> une bulle `aria-label="Message preview (not sent yet)"` apparait
   et contient le markdown rendu (balise `strong` ou `li`) ; effacer la fait
   disparaitre. Attendre le rendu par l'apparition de l'element, pas par un delai.
2. Majuscule/Entree : `Shift+Enter` insere un saut de ligne sans envoyer (le
   nombre de messages ne change pas).
3. Edit-and-resend : faire un envoi complet (une generation), puis cliquer
   `aria-label="Edit and resend"` -> le brouillon contient le texte d'origine, la
   conversation n'a plus que la bulle utilisateur supprimee (compter les cartes),
   puis un deuxieme envoi part avec le texte modifie. (Total : 2 generations.)
4. Redemarrage : `aria-label="Restart conversation"` -> conversation vide,
   l'illustration "Chat with the model" revient, et le bouton est `disabled`
   ensuite.
5. Pastilles de memoire : `GET /api/memory` renvoie 9 creneaux ; un clic sur une
   pastille avec du contenu l'ajoute au brouillon **sans ecraser** ce qui est deja
   ecrit. Si les creneaux sont vides (fichier par defaut), verifier seulement le
   nombre de pastilles et que le dialog de configuration s'ouvre.
6. Copie de la conversation : `aria-label="Copy conversation as markdown"` ->
   l'`aria-label` devient "Conversation copied" et le `title` "Copied" pendant
   ~1,5 s (ne pas assertir apres) ; le bouton est `disabled` quand la
   conversation est vide. Accorder la permission clipboard dans le contexte de
   test (`context.grantPermissions(['clipboard-read','clipboard-write'])`).

## Pieges

- Ne pas toucher au contenu du fichier `resources/memory_registration.json` : le
  test 5 ne poste pas de memoire, ou alors sur le serveur de fixture.
- `Edit and resend` n'existe que sur la derniere bulle utilisateur et a l'etat
  idle : attendre la fin du tour avant de le chercher.

## Acceptation

`npx playwright test e2e/11-chat-composer.spec.js` au vert.
