# Eye & Gesture Control 👁️✋

Une application futuriste qui vous permet de contrôler votre PC (souris, scroll, raccourcis) uniquement avec **vos yeux** et **vos mains**.

Conçu pour une configuration multi-écrans (4 moniteurs), ce projet utilise l'intelligence artificielle (MediaPipe) pour détecter où vous regardez et interpréter vos gestes en temps réel.

## 🚀 Fonctionnalités

### 1. Suivi du Regard (Eye Tracking)
L'application détecte l'orientation de votre tête pour savoir quel écran vous regardez.
*   **Changement d'écran automatique** : La souris saute instantanément sur l'écran que vous fixez.
*   **Support Multi-Moniteurs** : Gère jusqu'à 4 écrans (Gauche, Centre, Haut, Droite).

### 2. Contrôle Gestuel (Hand Tracking)
Des gestes spécifiques déclenchent des actions différentes selon l'écran actif :

| Écran | Geste | Action |
| :--- | :--- | :--- |
| **Tous** | ☝️ **Index levé** | **Déplacer la souris** (Suit le bout de l'index) |
| **Écran 1 (Web)** | ✌️ **Index + Majeur** | **Scroll HAUT** (Monter dans la page) |
| **Écran 1 (Web)** | 🤏 **Pince (Pouce+Index)** | **Scroll BAS** (Descendre dans la page) |
| **Écran 3 (Films)** | ✊ **Poing fermé** | **PAUSE / PLAY** (Active la fenêtre et appuie sur Espace) |

## 🛠️ Installation

### Prérequis
*   **Python 3.12** (Recommandé)
*   Une Webcam
*   Windows 10/11

### Étapes
1.  **Cloner ou télécharger** ce dossier.
2.  Ouvrir un terminal dans le dossier du projet.
3.  Créer un environnement virtuel (optionnel mais recommandé) :
    ```cmd
    py -3.12 -m venv venv
    venv\Scripts\activate
    ```
4.  Installer les dépendances :
    ```cmd
    pip install -r requirements.txt
    ```
    *Note : Si `pygetwindow` pose problème, assurez-vous d'avoir installé les outils de build C++ ou essayez `pip install pygetwindow --no-deps`.*

5.  **Télécharger les modèles IA** (si ce n'est pas déjà fait) :
    *   Les fichiers `face_landmarker.task` et `hand_landmarker.task` doivent être présents dans le dossier `eye_gesture_control`.

## 🎮 Utilisation

1.  Lancez l'application :
    ```cmd
    python eye_gesture_control/main.py
    ```
2.  Une fenêtre s'ouvre montrant le retour caméra.
3.  **Calibration** :
    *   Si la détection d'écran est imprécise, ouvrez `main.py`.
    *   Modifiez les valeurs dans `SCREEN_CENTERS` avec les valeurs `Yaw` et `Pitch` affichées sur votre écran quand vous regardez le centre de chaque moniteur.

4.  Pour quitter, appuyez sur la touche `Echap` (Esc) ou fermez la fenêtre.

## ⚙️ Configuration Avancée (`main.py`)

*   **`SCREEN_CENTERS`** : Coordonnées (Yaw, Pitch) du centre de vos écrans.
*   **`monitor_mapping`** : Correspondance entre vos écrans logiques (Gauche, Centre...) et les numéros de moniteurs Windows (0, 1, 2...).
*   **`SMOOTHING_FACTOR`** : Ajuste la fluidité de la souris (plus bas = plus fluide mais plus de latence).
*   **`SCROLL_SPEED`** : Vitesse de défilement.

## 📦 Dépendances Clés
*   `mediapipe` : Détection Visage et Mains (Google).
*   `opencv-python` : Traitement d'image.
*   `pyautogui` : Contrôle souris/clavier.
*   `screeninfo` : Détection des moniteurs physiques.
*   `pygetwindow` : Gestion des fenêtres (Focus).
