# test2e2-16 - Panels MCP/LSP de conversation : affichage, bascules, verrou

Niveau : etat d'UI avec `page.route` sur `GET /api/mcp` et `GET /api/lsp` (le
serveur sous test tourne avec les hubs desactives : c'est la seule fagon
d'observer ces panneaux sans demarrer un connecteur).
Spec : `web/e2e/16-chat-hub-panels.spec.js`.

## Cas d'usage couvert

Avant de parler, l'utilisateur choisit, par conversation, quels connecteurs et
serveurs de langage ont le droit d'intervenir. Ces choix geles une fois le
premier message accepte.

## A lire avant d'ecrire

- `web/src/components/WorkBenchMcpPanel.vue` et `WorkBenchLspPanel.vue` (miroirs
  l'un de l'autre : `visible`, `locked`, `canRestart`, le switch
  `role=switch`, le point de statut, le bouton "Restart … hub")
- `useChat.js` : `loadChatMcp`/`loadChatLsp` (filtre `active`, choix precedents
  conserves), `setChatMcp`/`setChatLsp` (refuses si `locked` ou ocuppe), le gel
  `state.mcp.locked = true` apres un envoi accepte, `restartHub`
- Le payload attendu de `GET /api/mcp` : `{installed, connectors:[{id,label,
  active,status}], maxIterations, externalsError}` et son jumeau `/api/lsp`
  (`{installed, servers:[...], maxIterations, serversError}`)

## Tests

1. Panneaux caches quand rien n'est actif : reponse stubbee avec
   `connectors: []` -> `aside[aria-label="MCP connectors for this
   conversation"]` absent (et idem LSP). La colonne de bascules ne prend pas de
   place.
2. Panneaux visibles et tous ON : reponse avec 2 connecteurs `active: true`
   (statut `running`) -> deux `role=switch` coches, `aria-label` "MCP <label> for
   this conversation", point de statut vert ; un connecteur `status: 'stopped'`
   est grise et son `title` dit que ses outils sont indisponibles.
3. Bascule possible a l'etat idle : decocher un switch le eteint (`state` relu
   par le switch) ; le `title` du switch explique ce qu'il fait.
4. Gel apres envoi : envoyer un message (1 generation reelle) -> les deux panneaux
   ont leurs switches `disabled` et le `title` dit "Toggles apply to the next
   conversation". Le bouton "Restart MCP hub" est `disabled` pendant le stream.
5. Restart debloque : `aria-label="Restart conversation"` -> switches de nouveau
   modifiables (le restart re-snapshot les panneaux).

## Pieges

- Stubbber les deux GET ne stubbe aucune generation : le test 4 envoie un vrai
  message, c'est autorise (1 seule generation pour la spec).
- Le gel est partage entre MCP et LSP : les deux panneaux doivent reagir
  ensemble, c'est l'invariant a verifier.

## Acceptation

`npx playwright test e2e/16-chat-hub-panels.spec.js` au vert.
