# test2e2-14 - WorkBench : mesures affichees (tok/s, contexte, chronometre)

Niveau : stream reel (une ou deux generations).
Spec : `web/e2e/14-chat-metrics.spec.js`.

## Cas d'usage couvert

Pendant une reponse, l'utilisateur suit la vitesse de generation, l'occupation du
contexte et la duree de la session - les trois badges du panneau lateral.

## A lire avant d'ecrire

- `web/src/components/WorkBenchSidePanel.vue` : les deux badges `role="status"`
  (debit `… t/s`, contexte `… ctx …`) et `ChatStopwatch`
- `web/src/components/ChatStopwatch.vue` (`aria-label="Elapsed time of the chat
  session"` pendant le tour, `aria-label="Chat session idle"` sinon, `role=timer`)
- `useChat.js` : `liveTokenRate`, `noteGenerated` / `closeGenerationWindow`, les
  frames `usage` (`contextTokens`, `tokensEstimated`)
- `web/src/components/GaugeBar.vue` : la jauge d'iterations d'outils du panneau
  d'outils (valeur/max issus de `MAX_TOOL_ITERATION`)

## Tests

1. Chronometre au repos : `role=timer` lit `00:00:00` et son `aria-label` vaut
   "Chat session idle".
2. Chronometre en tour : des l'envoi, l'`aria-label` devient "Elapsed time of the
   chat session" ; le texte n'est plus fige a `00:00:00`. A la fin du tour il
   repasse a `00:00:00` (l'arret est dans `finish()`).
3. Debit : le badge commence a `— t/s` (etat de repos) ; des que du texte est
   genere il presente le motif `\d+(\.\d+)? t/s` (eventuellement precede de `≈`).
   Assertir sur le motif, jamais sur la valeur.
4. Contexte : `— ctx` au repos ; apres une reponse terminee, le badge presente
   `ctx <nombre>` et son `title` est non vide. Une deuxieme lecture du badge ne
   doit pas le faire baisser.
5. Jauge d'outils : sans connecteur actif (flags `BENCHMARKIT_DISABLE_*`), le
   panneau `aria-label="Model reasoning and MCP tool calls"` ne montre aucun
   badge d'outil (`.chat-tool` absent) et sa jauge est a sa valeur de repos.

## Pieges

- Le debit est calcule cote client sur les chunks : sur une reponse tres courte il
  peut rester a `— t/s`. Demander une reponse longue (seule la longueur compte, pas
  le texte) et tolerer les deux cas tant que le motif est verifie des qu'il y a du
  texte.
- Deux generations au plus pour la spec entiere.
- Le nombre max d'iterations vient de `GET /api/mcp` (`maxIterations`) : ne pas
  l'ecrire en dur.

## Acceptation

`npx playwright test e2e/14-chat-metrics.spec.js` au vert.
