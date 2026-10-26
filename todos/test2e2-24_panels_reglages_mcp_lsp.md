# test2e2-24 - Reglages : sections MCP et LSP en lecture seule

Niveau : etat d'UI (serveur shared ; les hubs sont desactives mais leurs
declarations restent listees).
Spec : `web/e2e/24-settings-hubs.spec.js`.

## Cas d'usage couvert

L'utilisateur ouvre les sections MCP et LSP pour voir ce qui est declare, son
etat réel, et la limite d'iterations partagee - sans rien activer.

## Interdit ici

Aucun clic sur un interrupteur ni sur un bouton d'enregistrement : `POST
/api/mcp` et `POST /api/lsp` reecrivent `mcp_connectors/externals.json` et
`lsp_connectors/lsp_servers.json` (les fichiers de declaration reellement
utilises). Aucun `POST /api/mcp/restart` non plus : il relance les connecteurs.

## A lire avant d'ecrire

- `web/src/components/SettingsDialog.vue` : les sections `mcp` et `lsp` (liste,
  statut colore, texte de detail, le champ "Max tool iterations", l'erreur de
  validation affichee, l'indice d'installation quand le paquet manque)
- Payload attendu : `GET /api/mcp` -> `{installed, connectors:[{id,label,type,
  active,status,error,...}], maxIterations, externalsError}` ; `GET /api/lsp` ->
  `{installed, servers:[{id,label,extensions,workspaceRoot,...}], maxIterations,
  serversError}`. Les secrets (`headers`, `env`) ne sont **jamais** dans la
  reponse : c'est une assertion de la spec.
- `src/api/routes.py` : `_mcp_payload` / `_lsp_payload` (le statut vient du
  manager, `stopped` quand le hub ne s'est pas connecte)

## Tests

1. Coherence avec le serveur : les ids listés dans la section MCP sont exactement
   ceux de `GET /api/mcp` (fetch), et idem pour LSP. Les valeurs sensibles
   attendues (`ctx7sk-`, une cle d'API) n'apparaissent **ni** dans le payload,
   **ni** dans le texte de la boite de dialogue.
2. Statuts honnetes : avec les hubs desactives, chaque connecteur a le statut
   `stopped` et sa pastille grise ; aucun n'affiche "running".
3. Limite partagee : le champ "Max tool iterations" montre `maxIterations` du
   payload MCP, et la meme valeur dans la section LSP (un seul bouton pour les
   deux hubs).
4. Erreur de declaration (etat stubbe) : `page.route('/api/mcp')` renvoie
   `externalsError: "…"` -> le message est visible dans la section, et le reste
   de la liste utilisable. Justifier le stub : une vraie declaration invalide ne
   peut pas exister sans modifier le fichier de la machine.

## Acceptation

`npx playwright test e2e/24-settings-hubs.spec.js` au vert, et `git status` sans
modification de `mcp_connectors/externals.json` ni `lsp_connectors/lsp_servers.json`.
