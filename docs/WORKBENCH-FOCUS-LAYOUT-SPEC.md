# WorkBench — deux focalisations de l'écran de chat

**Statut** : spec écrite, implémentation non commencée.
**Surface** : l'onglet WorkBench uniquement. Rien côté backend, API ou SSE.

## 1. Le besoin

Pendant qu'un agent répond, la sortie de raisonnement (`WorkBenchToolPanel`) est
coincée dans la colonne gauche : large de `--bi-chat-tools-col-width` (2 ×
`--bi-chat-panel-width` de 15,5 rem + 0,5 rem, `theme.css:27-28`), et haute de
ce que laissent libre la barre du dock et le dock lui-même. Sur une réponse
longue, le raisonnement est la chose la plus intéressante à l'écran et c'est la
plus petite.

Symétriquement, quand on ne regarde que la conversation — relire un tour, taper
un message — les deux panneaux d'outils et la gauge occupent une bande
verticale pour rien.

On veut donc pouvoir mettre une des deux surfaces au premier plan, sans jamais
perdre le contrôle qui permet de revenir à la vue normale.

## 2. Les trois états

Un seul axe, trois valeurs. L'état appartient au composable, pas au CSS.

| État | Colonne gauche | Colonne droite | Gouttière |
|---|---|---|---|
| `normal` | en flux | en flux | visible |
| `reasoning` | `flex: 1 1 auto`, toute la place | `display: none` | visible |
| `conversation` | `display: none` | en flux | visible |

Les trois sont mutuellement exclusifs : une seule focalisation active à la fois,
sinon l'écran se vide. Un seul owner d'état les arbitre — pas deux refs
indépendantes qui peuvent valoir `true` ensemble.

État `reasoning`, colonne droite entièrement masquée (la « solution A ») :
conversation et composer sortent du flux. Il n'y a donc ni bouton Stop ni champ
de saisie à l'écran dans cet état : `reasoning` est une lecture, pas un poste de
pilotage. Pour interrompre une réponse en cours, on re-clique le contrôle de
focalisation — la colonne droite revient — puis on appuie sur Stop redevenu
visible. Deux temps, volontairement.

Alternatives rejetées :

- Conserver le composer en bas pendant l'état `reasoning`. Rejeté : l'état
  cesserait d'être « toute la place », et la hauteur du composer resterait
  gouvernée par le splitter `chatSplit` — deux barres actives simultanément sur
  des axes opposés.
- Un indicateur d'état de stream plus Échap pour interrompre. Rejeté : l'AppBar
  porte déjà l'indicateur d'activité, et un Échap qui interrompt alors qu'on
  attend un Échap qui réduit est un piège. Pas de raccourci clavier dans cette
  itération.

## 3. La gouttière : un troisième flex item

Les deux contrôles vivent dans une troisième colonne de largeur fixe, entre les
deux colonnes existantes, empilés verticalement — focalisation en haut,
masquage en dessous. Ce n'est ni un overlay ni un enfant du panneau de reasoning.

La raison est structurelle : `.chat-shell` est un `display: flex;
flex-direction: row; gap: var(--bi-space-2)` (`theme.css:1267-1275`) dont
`.chat-col--left` et `.chat-col--right` sont les deux seuls enfants
(`WorkBenchView.vue:78-109`). Un troisième enfant `flex: 0 0 auto` s'insère
entre eux sans que personne ait à recalculer quoi que ce soit.

C'est ce qui a écarté les deux autres placements. Un contrôle posé en
`position: absolute` avec un `left` dérivé de `--bi-chat-tools-col-width`
devrait être re-calé pour l'état `reasoning` (colonne pleine largeur), pour
l'état `conversation` (plus de colonne), pour le variant `chat-col--single`
(`theme.css:1293`, la colonne ne fait plus qu'une largeur de panneau) et pour
les deux breakpoints qui redéfinissent `--bi-chat-panel-width` à 11 rem puis
7 rem (`theme.css:2273-2283`). La gouttière, flex item indépendant, traverse
ces quatre situations sans les connaître.

Elle est aussi la seule surface qu'aucun des deux états ne masque. Les deux
contrôles sont donc toujours au même endroit, dans le même ordre, quel que soit
l'état. La règle « un contrôle doit survivre à l'état qu'il commande » — la
difficulté centrale des versions précédentes, et la raison pour laquelle le
chevron ne pouvait pas être posé sur la gauge bar (`chat-tool-gauge`,
`WorkBenchToolPanel.vue:179`) — devient inutile, donc inutile à vérifier.

Coût horizontal : un troisième enfant ajoute deux gap de 14 px, soit ~48 px
perdus pour la conversation. La gouttière annule ces deux gaps
(`margin-inline: calc(-1 * var(--bi-space-2))`) : le coût net tombe à ~28 px et
l'espacement visuel entre les colonnes ne change pas.

## 4. Le glyph, et son comportement au survol

Le contrôle de focalisation suit le dessin demandé :

- Repos (non focalisé) : un double carré — deux carrés concentriques, contours
  en pointillés.
- Survol : la marque « s'éclaire » — couleur et opacité passent au premier plan,
  les pointillés deviennent pleins. Pas d'animation de taille : la cible ne
  bouge pas sous le curseur.
- Focalisé : un carré simple en pointillés — on ne dessine pas le « restaurer »
  avec le même motif que le « maximiser ».

Le contrôle de masquage est un chevron pointillé du même trait, qui s'éclaire de
la même façon, et dont l'orientation indique ce qu'il va faire : chevron vers la
gauche quand la colonne gauche est visible (elle va se rabattre), vers la droite
quand elle est masquée (elle va revenir).

Les deux glyph sont des `<button type="button">` avec `aria-label` en anglais
comme le reste du chrome de l'app (« Hide the tool panels »,
`WorkBenchToolsDock.vue:56`) et `aria-pressed`, parce que ce sont des bascules
d'état et non des actions ponctuelles.

Zone cliquable : le glyph visible fait ~14 px, la cible au moins 28 px — la
gouttière est large de 28 px et chaque bouton occupe sa hauteur pleine. Le
pointillé est la marque, pas la cible.

Alternatives rejetées :

- Le chevron à gauche de la gauge bar. Rejeté, deux raisons : la gauge vit dans
  la colonne que l'état `conversation` masque, donc le contrôle qui permet
  d'entrer dans cet état disparaîtrait en y entrant, sans moyen d'en sortir ; et
  `.chat-tool-panel` est en `overflow: hidden` (`theme.css:2103`, nécessaire
  pour l'arrondi et la gauge), donc une marque voulue à moitié débordante y
  serait rognée.
- Le double carré dans l'angle bas-droit du panneau. Rejeté : cet angle est déjà
  occupé par la `GaugeBar` bottom-pinned (`theme.css:2116-2117`), dans le
  conteneur rogné ci-dessus.

## 5. Décisions de conception

### 5.1 Cacher en CSS, jamais en `v-if`

Les deux états masquent une colonne par `display: none` posé sur le shell,
jamais par `v-if` sur la colonne. Ce n'est pas une préférence de style, c'est
une contrainte du panneau de reasoning : ses lignes sont tenues dans un cache
local au composant, `rowCaches = new WeakMap()` (`WorkBenchToolPanel.vue:90`),
et son nombre de colonnes vient d'un `ResizeObserver` sur l'hôte de lignes
(`WorkBenchToolPanel.vue:109-110`). Un `v-if` détruirait le composant ; le
re-mount repartirait de zéro — cache vide, budget remis à `MIN_BUDGET`
(`reasoning-lines.js:24`, `TAB_WIDTH` = 4, `reasoning-lines.js:20`) et rempli
ligne par ligne — donc tout le raisonnement de la session serait re-découpé
sous les yeux de l'utilisateur, scroll perdu.

Conséquences positives de la règle : les refs de `useChatLayout` restent
valides (une ref Vue survit à `display: none`), les deux splitters ne sont pas
démontés, `useAutoScroll` (`useAutoScroll.js:8-24`, qui ne lit que
`scrollHeight`/`scrollTop` sur sa ref) continue de fonctionner sans rien
savoir, et le `ResizeObserver` n'est jamais déconnecté.

### 5.2 La géométrie persistée ne bouge pas

Aucun des trois états n'écrit dans `benchmarkit:chatToolsColumn` ni dans
`benchmarkit:chatSplit` (`useChatLayout.js:38,44`), et aucun ne lit leur valeur
pour décider d'une taille. Les états sortent des éléments du flux, ils ne
redimensionnent rien. Le retour à `normal` retrouve donc les hauteurs draguées
telles quelles — comportement déjà en place pour le pli du dock, dont le
commentaire est la référence pour ce principe : « Folding is a flag, not a
size: the dock's dragged height survives it untouched » (`useChatLayout.js:48-52`).

C'est aussi ce qui rend la gouttière gratuite à droite : le splitter du
composer prend `.chat-col--right` comme shell, voit sa colonne s'élargir toute
seule quand la gauche sort du flux, et n'a rien à faire.

### 5.3 Persistance : `conversation` oui, `reasoning` non

- `conversation` est une préférence d'aménagement : persistée sous
  `benchmarkit:chatLeftHidden`, sur le même pattern try/catch que `COLLAPSE_KEY`
  (`useChatLayout.js:26-34,53-64`). Elle est du même ordre que
  `benchmarkit:chatToolsCollapsed`.
- `reasoning` est un mode de lecture momentané : on y entre pour une réponse,
  on en sort pour reprendre la main. Le persister signifierait recharger l'app
  avec le composer absent — un écran sans moyen de parler au modèle.

Alternative rejetée : tout persister, ou ne rien persister. Dans les deux cas
soit on subit un écran muet au chargement, soit on reperd à chaque reload une
disposition qu'on vient de choisir.

### 5.4 Un seul axe, trois valeurs — pas deux booléens

`focus: 'normal' | 'reasoning' | 'conversation'` dans `useChatLayout`. Deux
booléens indépendants peuvent valoir `true` ensemble et produiraient un écran
vide ; un seul axe rend cet état inexprimable. L'exclusivité est dans le type,
pas dans une règle que le CSS devrait respecter.

### 5.5 Le dock MCP/LSP suit la colonne gauche

L'état `reasoning` sort aussi le dock et sa barre du flux, comme le fait déjà
`dockCollapsed` — sans toucher à la hauteur persistée de
`benchmarkit:chatToolsColumn`, donc le retour à `normal` retrouve le dock à sa
taille draguée. Sans cela, `reasoning` n'élargit que la colonne et laisse le
dock réclamer sa hauteur minimale (`DOCK_MIN = 104`, `useChatLayout.js:23`) :
le raisonnement gagnerait la largeur mais pas la hauteur, et le croquis
« maximisé » ne serait pas atteint.

### 5.6 L'état est porté par `.chat-shell`

Les classes d'état sont posées sur `.chat-shell` (`WorkBenchView.vue:78`), pas
sur les colonnes. C'est aussi le conteneur qui porte déjà les
`--bi-chat-panel-width` surchargés par les breakpoints (`theme.css:2275,2281`) :
l'état vit donc à l'endroit où la géométrie du shell se décide.

Noms de classes : `chat-shell--reasoning`, `chat-shell--conversation`. Le CSS
n'a que deux règles structurantes à ajouter (une par état), plus la gouttière.

## 6. Effets de bord connus, acceptés

Ils ne sont pas corrigés, parce qu'ils ne sont pas des défauts de ce
changement — mais celui qui implémente doit les connaître, et ne pas
« réparer » ce qui est un contrat déjà établi.

**Agrandir re-découpe tout le raisonnement affiché.** Le budget de lignes vient
d'un `ResizeObserver` ; `displayLines` teste `cache.budget !== width`
(`reasoning-lines.js:81`) et, sur changement, repart de zéro. Passer de `normal`
à `reasoning` change la largeur du panneau, donc le budget, donc toutes les
lignes sont re-coupées et la position de scroll perd son sens. C'est déjà le
comportement au passage d'un breakpoint : la règle « une ligne déjà rendue ne
bouge plus » vaut à largeur constante, pas entre deux largeurs. Un test pose
d'ailleurs ce contrat explicitement (`web/tests/reasoningLines.test.js:116`,
« re-wraps from scratch when the column budget changes »).

**Le budget tombe à son minimum pendant que la colonne est masquée.** En
`display: none`, l'hôte de mesure fait 0×0 : la sonde a une `getBoundingClientRect()`
de largeur 0, donc `advance > 0` est faux dans `remeasure()`
(`WorkBenchToolPanel.vue:73-75`), `columns = 0`, et `rowBudget` est ramené à
`Math.max(MIN_BUDGET, 0)`. Le raisonnement continue donc de se découper en 4
colonnes pendant que la colonne est cachée, et se re-découpe entièrement au
retour — cf. le point ci-dessus. Ce n'est pas cassé : le texte source
(`block.text`) reste la vérité, lue notamment par la sauvegarde de session.
Mais ce n'est pas gratuit : **ne pas chercher à préserver le budget à travers
un état masqué** — ce serait du code de plus pour un gain nul, le re-découpage
étant déjà inévitable de toute façon.

**L'état `conversation` masque la gauge de la boucle d'outils.** Cette gauge
(`chat-tool-gauge`, `WorkBenchToolPanel.vue:179`) montre les itérations
`MAX_TOOL_ITERATION` du tour en cours. En `conversation`, elle n'est plus
visible — c'est le but de l'état, pas une régression, et l'agent continue de
travailler. À accepter tel quel.

## 7. Pièces à toucher

Tout est frontend. Aucun fichier backend nouveau, aucune route, aucun event SSE.

| Fichier | Changement |
|---|---|
| `web/src/composables/useChatLayout.js` | Seul endroit où l'état est décidé : un `ref` `focus` à trois valeurs avec un setter unique qui arbitre l'exclusivité, et un `leftHidden` persisté sur le pattern de `COLLAPSE_KEY`. Les deux `useSplitter` ne bougent pas d'une ligne. Retourne `focus`, `setFocus`/`toggleFocus`, `leftHidden`, `toggleLeftHidden`. |
| `web/src/components/WorkBenchView.vue` | Deux choses : les classes d'état sur le `div.chat-shell` (ligne 78) qui reçoivent déjà `:class` sur la colonne gauche (ligne 82) — et un troisième enfant entre les deux colonnes, la gouttière, tenant les deux `<button>`. C'est le seul changement de structure du template. |
| `web/src/components/WorkBenchToolPanel.vue` | Rien, ou presque. Le panneau ne sait pas qu'il est focalisé : `flex: 1 1 0` (ligne 2096 du CSS) fait déjà tout. Au plus une classe pour l'accessibilité (`aria-pressed` rapporté). Si un changement y est nécessaire, c'est un signal que la géométrie est mal placée. |
| `web/src/styles/theme.css` | La gouttière (largeur 28 px, `flex: 0 0 auto`, `margin-inline` négatif, `align-self: flex-start`) ; les deux règles d'état sur `.chat-shell` ; le glyph (SVG en `stroke-dasharray` pour le pointillé, `currentColor`, et l'état `:hover`/`:focus-visible` qui « éclaire »). À insérer dans le bloc `/* -- chat screen */` (ligne 1261), pas à la fin du fichier. |

Ordre d'implémentation suggéré, chacun de ces pas étant vérifiable seul :

1. L'axe d'état dans `useChatLayout` (testable en isolation, sans CSS).
2. La gouttière visuelle en état `normal` seulement — les deux boutons ne font
   encore que changer l'état, rien ne bouge à l'écran.
3. Les règles CSS des deux états. À partir de là, la feature est fonctionnelle.
4. Le glyph et le survol.
5. Les tests.

Un point de doc à ne pas oublier : l'en-tête de `useChatLayout.js` (lignes 4-22)
**décrit la géométrie de l'écran** — le schéma des deux colonnes et la phrase
« Folding is the one control that spans both ». Ce commentaire est la source de
vérité locale pour celui qui touche la vue ensuite ; il doit gagner la
gouttière et les deux focalisations, ou il deviendra une affirmation fausse.
C'est là, et non dans `AGENTS.md`, que cette géométrie est décrite : `AGENTS.md`
ne mentionne pas le nombre de colonnes du chat et n'a donc pas à changer (il
décrit l'app en grand, et ses propres règles lui interdisent les enumerations
qui vieillissent mal).

## 8. Vérification

### Niveau 1 — unitaire / composant (`web/tests/`, Vitest + jsdom)

Le choix d'un seul axe d'état se vérifie là, sans navigateur, exactement comme
les autres composables testés de l'app (`useEditorWrite.test.js`, etc.). Nouveau
fichier `web/tests/useChatLayoutFocus.test.js` :

- les trois états sont atteignables et l'exclusivité tient : passer de
  `reasoning` à `conversation` ne laisse pas les deux actifs ;
- `leftHidden` survit à un re-mount (localStorage), `focus` non ;
- aucun des trois états n'écrit dans `benchmarkit:chatToolsColumn` ni
  `benchmarkit:chatSplit`, et le pli du dock (`chatToolsCollapsed`) reste
  indépendant des trois — on peut plier le dock dans n'importe quel état ;
- localStorage inaccessible (mode privé) : les bascules marchent quand même,
  comme pour `loadCollapsed()` (`useChatLayout.js:28-34`).

Un test de composant avec `@vue/test-utils` (déjà utilisé, ex.
`web/tests/useEditorWriteMatrix.test.js`) peut vérifier que les deux boutons
portent `aria-pressed` et un label, et que la gouttière est rendue dans les
trois états.

### Niveau 2 — e2e (`web/e2e/`, Playwright contre la vraie app)

C'est le seul niveau qui prouve que `display: none` fait ce qu'on attend — jsdom
ne fait pas de layout. Un `05-workbench-focus.spec.js`, dans le style des
specs existantes (elles utilisent `page.getByLabel(...)`, cf.
`01-app-boot.spec.js:32`) :

- en `reasoning`, `.chat-col--right` n'est pas visible (Playwright:
  `toBeHidden()`, ce qui est exactement le verdict pour `display: none`) et
  `.chat-tool-panel` a une largeur supérieure à sa largeur en état `normal` ;
- en `conversation`, `.chat-col--left` est cachée, et le contrôle qui a servi à
  masquer est **encore cliquable** — c'est l'assertion qui protège la règle du
  §3, celle que le placement sur la gauge bar aurait cassée ;
- la gouttière est visible dans les trois états ;
- un reload en `conversation` y revient, un reload en `reasoning` revient à
  `normal` (§5.3) ;
- une hauteur draguée du dock avant l'entrée en `reasoning` est retrouvée
  identique au retour (§5.2).

Ce que le niveau 2 ne vérifie pas : le re-découpage du raisonnement **pendant
un stream en cours** (§6). Le faire dépendre d'un tour d'agent réel rendrait le
spec non déterministe ; ce re-découpage est un comportement déjà établi et déjà
testé au niveau 1 sur le découpeur lui-même (`reasoningLines.test.js:116`).
Ce qui compte ici est qu'il est visible mais non corrompu : à vérifier une fois
à la main.

### Ce qui n'est pas testé, et pourquoi

L'aspect « double carré en pointillés qui s'éclaire » n'a pas de test : le CSS
est vérifié visuellement dans cette app (`vitest.config.js` — `css: false`,
« the theme is verified visually in e2e, not asserted here »). Ce qui est testé,
c'est la géométrie et l'état, pas le rendu.

## 9. Hors scope

- Le même mécanisme pour l'onglet Editor ou le terminal : rien de comparable
  n'existe aujourd'hui, et l'inverse n'est pas vrai non plus.
- Une focalisation de la **conversation seule** (composer visible, timeline
  pleine hauteur) : ce que l'état `conversation` donne déjà suffit, et ajouter
  un quatrième état rendrait l'exclusivité moins évidente à expliquer.
- Les raccourcis clavier.
- Rendre la gouttière dragable (largeur réglable) : sa largeur est un chiffre
  constant, et un troisième splitter dans une vue qui en a déjà deux demande
  une raison d'exister.
