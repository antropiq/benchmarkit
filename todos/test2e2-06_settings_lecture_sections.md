# test2e2-06 - Reglages : ouverture, sections, valeur lue, verrouillage

Niveau : etat d'UI (lecture seule ; aucune poste de reglages).
Spec : `web/e2e/06-settings-readonly.spec.js`.

## Cas d'usage couvert

L'utilisateur ouvre les reglages pour verifier la configuration courante sans
rien changer : les valeurs de `.env.test` doivent etre visibles, section par
section.

## A lire avant d'ecrire

- `web/src/components/SettingsDialog.vue` (les 7 sections : working-dir, llm,
  question-set, sampling, verifier, mcp, lsp), `ApiKeyField.vue`
- `useBenchmark.js` : `openSettings` / `closeSettings` / `saveSettings`
- `src/api/routes.py` : `GET /settings` (champs `serverUrl`, `endpoint`, `temp`,
  `questionSet`, `verifier.*`), `GET /path-location`

## Interdit dans cette spec

Aucun clic sur un bouton d'enregistrement (`POST /api/settings` reecrit le `.env`
du depot). Les valeurs attendues viennent des fetchs `GET /api/settings` et
`GET /api/path-location`, pas de chaines en dur.

## Tests

1. Ouverture et titre : bouton `aria-label="Open settings"` -> `dialog`
   `aria-labelledby="settings-title"` visible ; le message "Saved to the
   server-side `.env` file" est present.
2. Section LLM : les champs contiennent `serverUrl` et `endpoint` tels que
   renvoyes par `GET /api/settings`. Le champ cle d'API masque sa valeur
   (`type=password`) et le bouton `aria-label="Show API key"` le devoile.
3. Navigation entre sections : les 7 entrees de `nav[aria-label="Settings
   sections"]` sont presentes, `aria-current="true"` suit le clic.
4. Travailler dans la section "Working directory" : le champ Path affiche la
   valeur de `GET /api/path-location`, et le bouton `Set` est `disabled` tant que
   le champ n'a pas change.
5. Fermeture : `Escape` ou clic sur le fond du dialog le ferme, et aucun
   `POST /api/settings` n'a ete emis ( écoute `page.on('request')`).
6. Verrouillage : pendant un run, `aria-label="Settings (available when idle)"`
   est `disabled` et le dialog ne s'ouvre pas. Soit lancer un run puis verifier,
   soit stubber `GET /api/config` avec `runStatus: 'running'` : choisir une voie
   et la justifier en commentaire.

## Acceptation

`npx playwright test e2e/06-settings-readonly.spec.js` au vert, et
`git status` ne montre aucune modification de `.env`.
