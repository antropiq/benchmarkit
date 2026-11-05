# Spécification Fonctionnelle : Mode "Live View" pour le Panneau de Raisonnement

## 1. Contexte et Problématique

Le panneau de raisonnement (Reasoning Pane) dans l'onglet **WorkBench** accumule actuellement l'intégralité du flux de texte reçu via SSE. Sur les machines peu puissantes, la manipulation du DOM d'un texte très long entraîne une latence croissante et un ralentissement perceptible de l'interface.

## 2. Objectif

Ajouter une fonctionnalité permettant de limiter la mémoire utilisée par le panneau de raisonnement en ne conservant que la portion de texte visible à l'écran.

## 3. Fonctionnalités

### 3.1. Interface Utilisateur (UI)

- **Emplacement :** Dans la barre d'outils du panneau de raisonnement (WorkBench), sous les boutons existants (maximiser, réduire).
- **Élément :** Un bouton (icône suggérée : "Focus", "Align Bottom" ou "Eye").
- **État par défaut :** Actif (Mode "Live View" au chargement — le panneau borné
  est le plus économe ; le clic rend le mode "Historique Complet").

### 3.2. Logique du Bouton

Le bouton agit comme un interrupteur (Toggle) assujetti à l'état de l'inférence :

1. **État "Idle" (Repos) :**
   - Le bouton est **cliquable**.
   - Un survol (tooltip) indique : *"Activer le mode Live : l'historique non visible sera supprimé pour améliorer la performance."*

2. **Activation :**
   - Au clic, le mode "Live View" est activé.
   - Si du texte est déjà présent et visible, il est conservé. Si du texte est hors champ (au-dessus), il est supprimé.

3. **Pendant l'inférence (Streaming) :**
   - Le bouton devient **désactivé (disabled/locked)**.
   - L'utilisateur ne peut pas désactiver le mode en cours de route.
   - *Justification :* Garantir une expérience fluide et éviter des changements de comportement DOM brutaux pendant le flux.

4. **Fin de l'inférence (Retour à Idle) :**
   - Le bouton redevient **cliquable**.
   - L'utilisateur peut alors basculer vers le mode "Historique Complet" si désiré.

### 3.3. Comportement du Flux (Rendering)

Lorsque le mode "Live View" est actif :

- **Principe :** Le système ne conserve en mémoire (DOM) que les lignes visibles dans la fenêtre d'affichage (`viewport`).
- **Nettoyage (Pruning) :** Dès qu'une nouvelle ligne arrive et pousse une ligne existante hors du champ visible (vers le haut), cette ligne est supprimée du DOM.
- **Scroll Automatique :** Le panneau doit suivre le flux automatiquement (comportement `tail -f`). Le texte défilant reste toujours visible en bas du panneau.
- **Aucun ascenseur vertical :** tant que le mode Live est actif, le panneau ne présente jamais de barre de défilement verticale, même quand le flux dépasse sa hauteur. La boîte est coupée (clip), pas défilable : une barre n'offrirait qu'un geste capable d'emmener le lecteur hors du flux.

### 3.4. Impact sur la Sauvegarde de Session

- **Perte de données :** L'historique du raisonnement qui a défilé n'est plus stocké en mémoire.
- **Conséquence :** Si l'utilisateur tente de sauvegarder la session ou de copier l'historique complet, seules les dernières lignes visibles (ou une partie limitée) seront disponibles.
- **Acceptation :** Cette perte est assumée et acceptée par l'utilisateur en échange de la performance.

## 4. Contraintes Techniques

- **Frontend (Vue 3) :** La gestion du DOM doit être optimisée (ex: `MutationObserver` ou gestion explicite de la longueur du buffer) pour éviter les reflows inutiles.
- **État Réactif :** L'état du bouton doit être lié à l'état global du run (`isRunning` ou `status`).

## 5. Critères d'Acceptation

1. Le bouton est présent sous les autres contrôles du panneau.
2. Le bouton est grisé/inaccessible pendant qu'un run est en cours.
3. L'activation du mode permet de maintenir une interface fluide même sur une machine lente avec un long flux.
4. À la fin du run, le bouton est de nouveau utilisable.
5. En mode Live, aucun ascenseur vertical n'apparaît dans le panneau de raisonnement.
