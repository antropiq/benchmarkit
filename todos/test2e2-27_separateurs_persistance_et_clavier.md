# test2e2-27 - Separateurs de panneaux : glisser, clavier, persistance

Niveau : etat d'UI (aucune generation).
Spec : `web/e2e/27-splitters-accessibility.spec.js`.

## Cas d'usage couvert

L'utilisateur redimensionne la zone de saisie du chat, l'arbre de l'editeur et le
terminal. Son reglage survit a un rechargement, et tout est utilisable au clavier.

## A lire avant d'ecrire

- `web/src/composables/useSplitter.js` : `orientation` (la barre horizontale fait
  grandir le panneau **du bas**, la verticale celui de **gauche**), les minima
  `minResized`/`minOther`, `maxRatio`, la cle localStorage (`benchmarkit:chatSplit`,
  `benchmarkit:editorSidebar`, `benchmarkit:editorTerminal`), le double-clic et
  `Home` qui reviennent au defaut CSS, la touche pas a pas (`KEYBOARD_STEP`)
- Les trois barres : `role=separator` + leur `aria-label` ("Resize the message
  input…", "Resize the file tree…", "Resize the terminal…"), `tabindex="0"`
- `web/src/lib/tab-strip-wheel.js` (la molette fait defiler la rangee d'onglets de
   l'editeur) - verifiable par `scrollLeft`

## Tests

1. Glisser (chat) : mesurer la hauteur du panneau du bas, `dragTo` la barre vers
   le haut, re-mesurer -> la taille a change dans le sens attendu, la cle
   `benchmarkit:chatSplit` contient un entier positif.
2. Clavier : focus sur la barre du chat (`role=separator`,
   `aria-orientation="horizontal"`) -> `ArrowUp` agrandit le panneau du bas,
   `ArrowDown` le retrecit (c'est le sens defini par `onKeyDown`, pas l'axe de la
   barre) ; chaque frappe ecrit dans localStorage. `Home` remet a zero : la cle
   est supprimee et la hauteur revient au defaut CSS. Le separateur n'expose pas
   `aria-valuenow` : passer par la mesure du panneau.
3. Bornes : tirer la barre tres au-dela du panneau oppose -> la hauteur du panneau
   du bas ne descend pas sous son minimum et le panneau du haut garde une hauteur
   non nulle (les deux `clamp` du composable).
4. Editeur : les deux barres (`vertical` et `horizontal`) reagissent, chacune ecrit
   dans **sa** cle ; apres `page.reload()`, les reglages sont restaures (la
   hauteur de la bande terminal est celle laisse).
5. Rangement : `test.afterEach` vide les trois cles localStorage pour ne pas
   imposer un reglage aux specs suivantes.

## Pieges

- Les minima du composable (px) dependent de la taille de la fenetre (1440x900
  fixee par la config) : assertir sur un mouvement et une bornes, pas sur un
  nombre de pixels exact.
- Le drag se fait par `page.mouse` (pointer events) : un `dragTo` sur l'element
  `role=separator` suffit si la barre est assez large, sinon `mouse.down/move/up`.

## Acceptation

`npx playwright test e2e/27-splitters-accessibility.spec.js` au vert.
