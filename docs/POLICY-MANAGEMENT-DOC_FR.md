# Politique de l'espace de travail — guide utilisateur

Comment benchmarkit décide quels fichiers d'un projet il peut afficher, lire et
modifier, et comment configurer cela pour **votre** projet.

Rien ici ne parle du fonctionnement interne de benchmarkit : tout tient dans un
petit fichier JSON posé dans le dossier que vous indiquez à l'application.

---

## 1. Le principe en un paragraphe

Le **dossier de travail** est le dossier que l'onglet Éditeur parcourt et modifie.
Benchmarkit pose une seule question à chaque fichier qu'il contient : *puis-je
l'afficher, le lire, le modifier ?* La réponse vient d'un fichier qui vous
appartient :

```text
<dossier de travail>/.nojae/read-write-policy.json
```

Trois réponses sont possibles, appelées *modes* :

| mode | dans l'arborescence | peut être ouvert / lu | peut être modifié (enregistrement, création, renommage, suppression) |
|---|---|---|---|
| `rw` | oui | oui | oui — c'est le défaut de tout ce qu'aucune règle ne mentionne |
| `ro` | oui | oui | **non** |
| `hidden` | **non** | **non** — il semble ne pas exister | **non** |

La politique s'applique aussi aux outils de fichiers utilisés par l'assistant de
onglet Chat : un dossier marqué `ro` n'est pas modifiable par un agent non plus,
et un dossier `hidden` lui est invisible. Un seul fichier, une seule réponse,
quel que soit celui qui demande.

## 2. D'où vient ce fichier

Quand un dossier devient votre dossier de travail — au démarrage de
l'application, ou quand vous le changez dans **Réglages** — benchmarkit y écrit
une politique toute prête **s'il n'en existe pas déjà**. Ce fichier par défaut
masque le bruit habituel (`.git`, les caches) et rend les dossiers de dépendances
et de construction accessibles en lecture : vous partez d'un exemple concret
plutôt que d'un fichier vide.

Deux choses à savoir sur cette initialisation :

* elle **n'écrase jamais** un fichier que vous avez modifié — une politique
  existante est laissée telle quelle ;
* si le dossier ne peut pas recevoir le fichier (montage en lecture seule,
  propriétaire différent), l'application vous le signale et ouvre quand même le
  projet ; vous pouvez créer le fichier vous-même.

Benchmarkit ne modifie plus jamais ce fichier, et aucune interface de
l'application ne permet de le faire. **Vous l'éditez en dehors de
l'application**, dans l'éditeur de votre choix, et les nouvelles règles
s'appliquent dès la navigation suivante — sans redémarrage.

Le dossier `.nojae` est réservé par benchmarkit : il n'apparaît jamais dans
l'arborescence et aucune route ne peut le lire ni y écrire. C'est pourquoi le
fichier s'édite hors de l'application, et pourquoi aucune règle de ce fichier ne
peut mentionner `.nojae`.

Il est aussi **local à la machine** : `.nojae/` n'est pas sous contrôle de
version, chaque copie du projet a donc la sienne. Le tenir hors de git est
voulu : les règles sensées d'un poste de développement et celles d'un serveur ne
se recouvrent pas.

## 3. Le fichier

```json
{
  "version": 1,
  "readme": [
    "benchmarkit reads this file when it browses this folder.",
    "Rules are matched in order and the LAST match wins."
  ],
  "policies": [
    { "path": ".git",                         "mode": "hidden" },
    { "path": "**/__pycache__",               "mode": "hidden" },
    { "path": "**/node_modules",              "mode": "ro",
      "reason": "browse the docs, edit nothing" },
    { "path": "**/node_modules/**/README.md", "mode": "rw" }
  ]
}
```

| clé | obligatoire | ce que c'est |
|---|---|---|
| `version` | oui | doit valoir `1` |
| `readme` | non | du texte pour vous. Le JSON n'a pas de commentaires, c'est là qu'ils vivent. Jamais comparé aux chemins |
| `policies` | oui | la liste des règles, dans l'ordre où elles s'appliquent |

Une règle comporte trois champs :

| champ | obligatoire | ce que c'est |
|---|---|---|
| `path` | oui | le motif, relatif au dossier de travail |
| `mode` | oui | `hidden`, `ro` ou `rw` |
| `reason` | non | vos mots à vous. Ils vous sont montrés (et à l'assistant) dès que c'est cette règle qui a bloqué quelque chose |

Le champ `reason` mérite d'être utilisé : un refus qui dit *« browse the docs,
edit nothing »* s'explique tout seul six mois plus tard, alors que *« ro »* ne
dit rien.

## 4. Les motifs — ce qui correspond vraiment

Les règles se comparent au chemin **à l'intérieur du dossier de travail** : le
fichier `web/src/api/server.js` d'un projet `/home/vous/proj` s'écrit
`web/src/api/server.js` — jamais le chemin absolu, jamais un chemin préfixé par
le nom du projet.

| écriture | signifie |
|---|---|
| `web` | ce dossier — **et tout ce qu'il contient** |
| `web/src` | ce dossier et tout son contenu |
| `*.md` | tout fichier `.md` du premier niveau seulement |
| `**/*.md` | tout fichier `.md`, à n'importe quel niveau |
| `config?` | `configa`, `config1` — `?` est un caractère |
| `.env*` | les noms *commençant* par `.env` : `.env`, `.env.local` |
| `**/node_modules` | `node_modules` à la racine **ou à n'importe quelle profondeur** |
| `web/**/dist` | tout dossier `dist` sous `web`, à n'importe quelle profondeur |

Quatre points qui surprennent le plus souvent :

1. **`*` ne traverse jamais un répertoire.** `web/*` couvre `web/index.html` mais
   **pas** `web/src/api/server.js`. Pour couvrir une arborescence, écrivez le nom
   du dossier seul (`web`) ou `web/**`.
2. **Une règle qui nomme un dossier couvre tout son contenu.** `.git` suffit à
   masquer `.git/config`, `.git/objects/…` et tout le reste en dessous.
3. **Les noms sont sensibles à la casse.** `dist` et `Dist` sont deux dossiers
   différents au niveau des règles : écrivez le nom exactement comme le dossier
   l'orthographie, et sur un système insensible à la casse, gardez une seule
   orthographe.
4. **Une règle ne peut pas sortir du dossier de travail.** Les chemins absolus
   (`/etc`), `~`, `..` et le `/` final sont refusés en tant qu'erreurs — vous ne
   pouvez pas protéger un fichier situé hors du dossier indiqué, et ce n'est pas
   nécessaire : le dossier de travail est déjà cette frontière.

## 5. L'ordre compte : la dernière correspondance gagne

Les règles se lisent de haut en bas et la **dernière** qui correspond décide.
C'est ainsi que l'on ménage une exception dans une règle large :

```json
{ "path": "**/node_modules",              "mode": "ro" },
{ "path": "**/node_modules/**/README.md", "mode": "rw" }
```

La seconde ligne ne fonctionne que parce qu'elle vient **après** la première.
Placez-la avant, et elle ne fait plus rien.

La seule exception est `hidden` : un dossier masqué est coupé entièrement, rien
en dessous ne peut être récupéré. Une exception écrite dans un dossier `hidden`
est ignorée — un `ro` ou un `rw` postérieur ne peut rien sauver de là-dessous —
et benchmarkit vous le signale par une *note* plutôt que par une erreur. Pour
masquer l'essentiel d'un dossier tout en le gardant parcourable, mettez `ro` sur
le dossier et `hidden` sur ce que vous voulez faire disparaître :

```json
{ "path": "secrets",            "mode": "ro" },
{ "path": "secrets/tokens",     "mode": "hidden" },
{ "path": "secrets/README.md",  "mode": "rw" }
```

## 6. Recettes

**Garder un dossier de bibliothèques tierces consultable mais intouchable**

```json
{ "path": "third_party", "mode": "ro" }
```

**Garder l'arbre de construction consultable, mais cacher le gros de sa sortie**

```json
{ "path": "**/build",        "mode": "ro" },
{ "path": "**/build/assets", "mode": "hidden" }
```

**Protéger une zone entière, puis rendre un fichier**

```json
{ "path": "db/migrations",           "mode": "ro" },
{ "path": "db/migrations/README.md", "mode": "rw" }
```

**Verrouiller du code généré, qui revient de toute façon du générateur**

```json
{ "path": "src/generated/**", "mode": "ro",
  "reason": "regenerated by make codegen, hand edits are lost" }
```

**Protéger un dossier qui n'existe pas encore**

```json
{ "path": ".aws", "mode": "hidden" }
```

Une règle concernant un dossier absent est parfaitement légale et attend son
apparition.

## 7. Quand quelque chose ne va pas

Il y a deux sortes de remarques, et elles ne se comportent pas pareil.

**Les erreurs** — le fichier n'est pas compris : une règle mal écrite, un `mode`
mal orthographié, `"policies"` devenu `"policyies"`, un `version` manquant, un
JSON invalide. Dans ce cas le fichier n'apporte **aucune règle**, et **rien dans
le dossier de travail ne peut être modifié** tant qu'il n'est pas corrigé — non
par sévérité, mais parce que les règles qui devaient protéger un dossier ne se
sont visiblement pas chargées. La lecture et la navigation restent disponibles
pour que vous puissiez voir le problème. Le message nomme le fichier et la règle.

**Les notes** — le fichier est bon, une règle ne sera simplement jamais entendue.
Une note est une information, pas un mur ; le reste du fichier fonctionne.

Ce que vous rencontrerez probablement, ce que cela veut dire et quoi faire :

| ce que dit benchmarkit | ce que cela veut dire | quoi faire |
|---|---|---|
| *the workspace policy did not load, so nothing in this workspace can be changed until it does* | une erreur n'importe où dans le fichier désactive toute écriture du projet | corrigez cette ligne ; le fichier est nommé en entier |
| *is not valid JSON: … (line N, column M)* | une virgule, un guillemet, une virgule finale | ouvrez le fichier à cette ligne |
| *has an unknown key "policis"* | une clé de premier niveau est mal orthographiée, vos règles n'ont jamais été lues | corrigez l'orthographe — une clé que personne ne lit n'est pas l'absence de clé |
| *mode "readonly" is not one of "hidden", "ro", "rw"* | le mode s'écrit exactement ainsi | écrivez `ro` |
| *policies[3] path "web/node_module" can never match: `web` has no "node_module" — did you mean "web/node_modules"?* | faute de frappe dans un nom de dossier | suivez la suggestion ; `[3]` est la position de la règle dans la liste `policies`, en comptant depuis 0 |
| *path ".nojae" names the folder this file lives in, which benchmarkit reserves* | aucune règle ne peut mentionner le dossier qui contient les règles | supprimez cette règle |
| *it is a bare "\*\*", which would apply to the whole workspace* | une règle ne peut pas porter sur le dossier de travail lui-même | écrivez `**/*`, ou nommez les dossiers |
| *it is absolute: a rule is written relative to the workspace* | les motifs ne commencent jamais par `/` | retirez le chemin de tête |
| *policies[5] sits inside the hidden rule at policies[1]* | note : cette exception ne peut rien faire revivre dans un dossier masqué | passez la règle parente en `ro`, ou supprimez l'exception |
| *policies[2] repeats policies[0]* | note : deux règles identiques, seule la dernière est citée | gardez-en une |

Les numéros entre crochets comptent les entrées de la liste `policies` à partir
de **0**, dans l'ordre où elles sont écrites.

## 8. Pièges, rassemblés

1. **Un fichier cassé coûte tout le projet, pas un dossier.** Une virgule en trop
   fait refuser chaque enregistrement, création, renommage et suppression du
   dossier de travail. Après avoir édité le fichier, tentez une écriture sans
   risque (ou simplement parcourez) : si l'arborescence répond et que les
   écritures passent, le fichier est compris.
2. **Les règles ne sont pas supprimées, elles sont ignorées** — tant que le
   fichier ne se charge pas. Rien ne se voit dans l'arborescence, ce qui rend le
   message « les modifications sont désactivées » facile à manquer.
3. **`*` n'est pas `**`.** L'erreur la plus fréquente, et elle masque ou verrouille
   bien plus que prévu.
4. **Une règle sur un dossier est une règle sur tout son contenu.**
   `{"path": "web", "mode": "hidden"}` masque tout le frontend. Il n'existe aucun
   moyen de masquer un dossier en gardant son contenu visible, et c'est voulu.
5. **Une exception dans un dossier `hidden` ne fait rien.** Mettez `ro` sur le
   parent (§5).
6. **La dernière gagne — sauf pour `hidden`.** Quand deux règles de forces
   différentes se recouvrent, celle qui est plus bas dans la liste répond ; mais
   une correspondance `hidden` l'emporte toujours sur une règle plus permissive,
   quelle que soit sa position. Règles larges d'abord, exceptions ensuite.
7. **Une règle sur un dossier absent n'est pas une erreur** — mais une règle à un
   ou deux lettres d'un nom de dossier réel **l'est**. Si benchmarkit suggère un
   nom voisin, il a presque toujours raison.
8. **Vous ne pouvez pas déverrouiller ce que benchmarkit a verrouillé.** Dans le
   propre dossier de benchmarkit, quelques chemins sont protégés par
   l'application elle-même ; une règle de politique qui essaie de les assouplir
   reçoit une note et aucun effet. Dans tout autre projet, ce fichier est la loi
   entière, aussi une telle note n'apparaît jamais — et là un fichier comme
   `.env` est un fichier ordinaire : masquez-le vous-même si vous voulez qu'il le
   soit.
9. **Le fichier est propre à chaque copie.** Copier
   `.nojae/read-write-policy.json` dans un nouveau dépôt est une étape manuelle ;
   un dépôt frais reçoit le fichier par défaut.
10. **`hidden` n'est pas un coffre.** Cela masque un fichier de l'arborescence de
    l'Éditeur, de ses lectures et des outils de fichiers de l'assistant. Rien
    n'est chiffré : le fichier est toujours sur le disque, toujours dans git, et
    lisible par tous les programmes de la machine.
11. **Supprimer une règle par défaut a des conséquences.** Ces lignes sont la
    raison pour laquelle `node_modules` est consultable mais verrouillé et `.git`
    invisible ; retirez-en une et cette protection disparaît dès la navigation
    suivante.
12. **La modification du fichier prend effet immédiatement.** Rechargez
    l'arborescence : aucun redémarrage n'est nécessaire, et l'application ne
    réécrira pas le fichier dans votre dos.

## 9. Liste de contrôle pour un nouveau projet

1. Indiquez le projet comme dossier de travail (Réglages).
2. Ouvrez `<projet>/.nojae/read-write-policy.json` dans un éditeur externe.
3. Lisez les règles par défaut ; supprimez celles qui ne conviennent pas.
4. Ajoutez `ro` pour tout ce qui est généré ou tiers que vous voulez pouvoir lire.
5. Ajoutez `hidden` pour tout ce que vous ne voulez ni voir ni toucher.
6. Ajoutez un `reason` à toute règle dont vous oublierez le sens dans six mois.
7. Enregistrez, puis parcourez l'arborescence : la forme que vous voyez est la
   politique que vous avez écrite.
