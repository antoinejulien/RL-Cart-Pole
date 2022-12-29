# PROBLÈME DU CHARIOT ET DU BÂTON

Code python 3 servant à l'entraînment d'un tenseur par l'algorithme 
Q-Learning. Le but du projet est d'implémenter l'algorithme d'apprentissage
par renforcement et de le comparer à une approche physique et à la
performance humaine.
---
## Libraries nécessaires

- numpy
- matplotlib
- sklearn
- pygame
- keyboard

À noter que la libraire "keyboard" ne peut être exécuter qu'avec des
droits administrateur ("sudo") sur Linux et les autres librairies
doivent donc être installées sous les mêmes contraintes.
---
## Options principales

- --policy 
  - sélection de l'algorithme de prise de décisions
  - choix: ["Q", "physics", "human"]
    - le choix "Q" enregistrera deux tenseurs numpy dans le répertoire. Un au quart de l'entraînement (un agent intermédiaire) et un à la fin (un agent performant)
    - le choix "human" combiner à l'option --render_mode human permet à l'utilisateur de jouer avec le chariot à l'aide des flèches directionnelles du clavier.
  - défaut: "Q"

- --render_mode 
  - type d'observation des épisodes lors de l'apprentissage
  - choix: ["rgb_array", "human", "quick_human"]
    - rgb_array: +rapide, aucune fenêtre produite
    - human: interface réaliste, cohérente avec la physique du problème
    - quick_human: interface réaliste accélérée
  - défaut: "rgb_array"

- --test
  - permet d'utiliser un tenseur "Q_advanced.npy" pré-enregistrer dans le répertoire pour tester ses performances
  - aucun tenseur n'est effacé ou enregistré

- --seed
  - reproductibilité de l'expérimentation
  - choix: entier positif

---
## Exécution

#### Pour entraîner ou tester un nouvel agent

    python3 main.py [OPTIONS]...

#### Pour jouer
    
    sudo python3 main.py --policy human --render_mode human

#### Pour calculer l'énergie du système libre ou en l'application de force constante

    python3 main.py --policy ["none", "constant0", "constant1"] --delete_limits

Les constantes physiques du problème comme la gravité, la masse des objets et la magnitude de la force appliquée peuvent
être modifiées directement dans la fonction __init()__ du fichier cartpole.py