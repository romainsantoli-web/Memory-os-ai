
# Fichier : memory_os_ai_gui.py
# Description : Interface graphique pour Memory OS AI, un système d'IA avancé pour la gestion et l'analyse de documents localement.
# Copyright (c) 2025 Kocupyr Romain
# Licence : LGPL v3 pour usage non commercial ; licence commerciale payante pour usage commercial.

import sys
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QTextEdit, QLineEdit, QLabel, QGridLayout, 
                               QFrame, QScrollArea)
from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QFont, QColor, QPainter, QPen
import threading
from memory_os_ai import (charger_fichiers, ingerer_textes, resumer_fichiers, 
                          rechercher_occurrences_faiss, generate_dynamic_response, 
                          FICHIERS_SEGMENTS_MAP, FICHIERS_RESUMES, TEXTES_DOCUMENTS, 
                          INDEX_FAISS, DOSSIER_FICHIERS, encode_chunk_cached)

class DataFlowWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.offset = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#00b7eb"), 2, Qt.DashLine)
        painter.setPen(pen)
        for i in range(0, self.width(), 50):
            x = (i + self.offset) % self.width()
            painter.drawLine(x, 25, x + 30, 25)

    def update_animation(self):
        self.offset = (self.offset + 5) % 50
        self.update()

class FileIconWidget(QWidget):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.setFixedSize(100, 100)
        self.setStyleSheet("border: 2px solid #00b7eb; border-radius: 10px; background-color: #2a2a2a;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Exo", 10))
        painter.drawText(self.rect(), Qt.AlignCenter, self.filename)

    def mousePressEvent(self, event):
        self.parent().parent().select_file(self.filename)

class MemoryOSAIWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Memory OS AI")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #1a1a1a;")

        # Widget central
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # En-tête
        self.header_label = QLabel("Memory OS AI")
        self.header_label.setFont(QFont("Orbitron", 24, QFont.Bold))
        self.header_label.setStyleSheet("color: white; background-color: transparent;")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.header_label)

        # Message de licence
        self.license_label = QLabel("Ce logiciel est sous licence LGPL v3 pour une utilisation non commerciale. Pour un usage commercial, veuillez acheter une licence : romainsantoli@gmail.com")
        self.license_label.setFont(QFont("Exo", 12))
        self.license_label.setStyleSheet("color: #00b7eb; background-color: transparent;")
        self.license_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.license_label)

        # Message de don
        self.donate_label = QLabel("Soutenez Memory OS AI en faisant un don ! [Lien vers Patreon/Open Collective]")
        self.donate_label.setFont(QFont("Exo", 12))
        self.donate_label.setStyleSheet("color: #00b7eb; background-color: transparent;")
        self.donate_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.donate_label)

        # Message de support
        self.support_label = QLabel("Besoin d'aide ? Contactez-moi pour un support payant : romainsantoli@gmail.com")
        self.support_label.setFont(QFont("Exo", 12))
        self.support_label.setStyleSheet("color: #00b7eb; background-color: transparent;")
        self.support_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.support_label)

        # Barre de recherche
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Entrez votre requête...")
        self.search_bar.setFont(QFont("Exo", 12))
        self.search_bar.setStyleSheet("background-color: #2a2a2a; color: white; border: 2px solid #00b7eb; border-radius: 10px; padding: 5px;")
        self.main_layout.addWidget(self.search_bar)

        # Layout principal
        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        # Section gauche : Grille de fichiers
        self.files_frame = QFrame()
        self.files_frame.setStyleSheet("background-color: #2a2a2a; border: 2px solid #00b7eb; border-radius: 10px;")
        self.files_layout = QVBoxLayout(self.files_frame)
        self.files_label = QLabel("Fichiers Chargés")
        self.files_label.setFont(QFont("Exo", 14))
        self.files_label.setStyleSheet("color: #00b7eb; background-color: transparent;")
        self.files_layout.addWidget(self.files_label)
        self.files_scroll = QScrollArea()
        self.files_scroll.setWidgetResizable(True)
        self.files_container = QWidget()
        self.files_grid = QGridLayout(self.files_container)
        self.files_scroll.setWidget(self.files_container)
        self.files_layout.addWidget(self.files_scroll)
        self.content_layout.addWidget(self.files_frame, 2)

        # Section centrale : Résultats
        self.results_frame = QFrame()
        self.results_frame.setStyleSheet("background-color: rgba(42, 42, 42, 200); border: 2px solid #00b7eb; border-radius: 10px;")
        self.results_layout = QVBoxLayout(self.results_frame)
        self.results_label = QLabel("Résultats")
        self.results_label.setFont(QFont("Exo", 14))
        self.results_label.setStyleSheet("color: #00b7eb; background-color: transparent;")
        self.results_layout.addWidget(self.results_label)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Exo", 12))
        self.results_text.setStyleSheet("background-color: transparent; color: white; border: none;")
        self.results_layout.addWidget(self.results_text)
        self.content_layout.addWidget(self.results_frame, 5)

        # Section droite : Contrôles et noyau
        self.controls_frame = QFrame()
        self.controls_frame.setStyleSheet("background-color: #2a2a2a; border: 2px solid #00b7eb; border-radius: 10px;")
        self.controls_layout = QVBoxLayout(self.controls_frame)
        self.controls_label = QLabel("Contrôles")
        self.controls_label.setFont(QFont("Exo", 14))
        self.controls_label.setStyleSheet("color: #00b7eb; background-color: transparent;")
        self.controls_layout.addWidget(self.controls_label)

        # Boutons
        self.load_button = QPushButton("Charger Fichiers")
        self.load_button.setFont(QFont("Exo", 12))
        self.load_button.setStyleSheet("background-color: #00b7eb; color: black; border-radius: 10px; padding: 10px;")
        self.load_button.clicked.connect(self.charger_fichiers)
        self.controls_layout.addWidget(self.load_button)

        self.search_button = QPushButton("Rechercher")
        self.search_button.setFont(QFont("Exo", 12))
        self.search_button.setStyleSheet("background-color: #00b7eb; color: black; border-radius: 10px; padding: 10px;")
        self.search_button.clicked.connect(self.rechercher)
        self.controls_layout.addWidget(self.search_button)

        self.report_button = QPushButton("Générer Rapport")
        self.report_button.setFont(QFont("Exo", 12))
        self.report_button.setStyleSheet("background-color: #00b7eb; color: black; border-radius: 10px; padding: 10px;")
        self.report_button.clicked.connect(self.generer_rapport)
        self.controls_layout.addWidget(self.report_button)

        self.open_button = QPushButton("Ouvrir Fichier")
        self.open_button.setFont(QFont("Exo", 12))
        self.open_button.setStyleSheet("background-color: #00b7eb; color: black; border-radius: 10px; padding: 10px;")
        self.open_button.clicked.connect(self.ouvrir_fichier)
        self.controls_layout.addWidget(self.open_button)

        # Noyau "Semantic Search Engine"
        self.core_label = QLabel("Semantic Search Engine")
        self.core_label.setFont(QFont("Exo", 12))
        self.core_label.setStyleSheet("color: white; background-color: #00b7eb; border: 2px solid #00b7eb; border-radius: 10px; padding: 10px; text-align: center;")
        self.core_label.setAlignment(Qt.AlignCenter)
        self.controls_layout.addWidget(self.core_label)

        self.content_layout.addWidget(self.controls_frame, 2)

        # Animation de flux
        self.data_flow = DataFlowWidget()
        self.main_layout.addWidget(self.data_flow)

        # Variables de l'application
        self.fichiers_segments = {}
        self.selected_file = None

    def charger_fichiers(self):
        self.results_text.setText("Chargement des fichiers...")
        threading.Thread(target=self._charger_fichiers_thread).start()

    def _charger_fichiers_thread(self):
        textes_fichiers, fichiers_segments, total_pages = charger_fichiers(DOSSIER_FICHIERS)
        embeddings, fichiers_segments = ingerer_textes(textes_fichiers, fichiers_segments, total_pages)
        resumer_fichiers(fichiers_segments)
        self.fichiers_segments = fichiers_segments
        self.update_files_grid()

    def update_files_grid(self):
        for i in reversed(range(self.files_grid.count())):
            self.files_grid.itemAt(i).widget().setParent(None)
        row, col = 0, 0
        for fichier in self.fichiers_segments.keys():
            icon = FileIconWidget(fichier, self)
            self.files_grid.addWidget(icon, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        self.results_text.setText("Fichiers chargés avec succès !")

    def select_file(self, filename):
        self.selected_file = filename
        self.results_text.setText(f"Fichier sélectionné : {filename}\nRésumé : {FICHIERS_RESUMES.get(filename, 'Aucun résumé disponible.')}")

    def rechercher(self):
        query = self.search_bar.text().strip()
        if not query:
            self.results_text.setText("Veuillez entrer une requête.")
            return
        threading.Thread(target=self._rechercher_thread, args=(query,)).start()

    def _rechercher_thread(self, query):
        total_occurrences, resultats = rechercher_occurrences_faiss(query, self.fichiers_segments)
        if resultats:
            resumes_pertinents = {fichier: FICHIERS_RESUMES[fichier] for fichier in resultats.keys()}
            prompt = f"[INST] Réponds en français uniquement. Fournis une réponse claire et bien présentée en français, basée sur les résultats suivants, sans mentionner 'segments'. Indique le nombre total d’occurrences du mot-clé '{query}', liste les fichiers avec leur nombre d’occurrences, triés par pertinence (nombre d’occurrences décroissant), et résume brièvement chaque fichier où le mot-clé apparaît :\nRésultats : Total = {total_occurrences}, Fichiers = {resultats}, Résumés = {resumes_pertinents}\nRéponse (en français uniquement) : [/INST]"
            reponse = generate_dynamic_response(prompt)
            self.results_text.setText(reponse)
        else:
            self.results_text.setText(f"Aucune occurrence du mot-clé '{query}' n’a été trouvée dans les fichiers.")

    def generer_rapport(self):
        sujet = self.search_bar.text().strip()
        if not sujet:
            self.results_text.setText("Veuillez entrer un sujet pour le rapport.")
            return
        threading.Thread(target=self._generer_rapport_thread, args=(sujet,)).start()

    def _generer_rapport_thread(self, sujet):
        emb_requete = encode_chunk_cached(tuple([sujet]))[0]
        distances, indices = INDEX_FAISS.search(np.array([emb_requete]), k=200)
        contexte_general = "\n".join([TEXTES_DOCUMENTS[idx] for idx in indices[0] if idx < len(TEXTES_DOCUMENTS)])[:1500]
        prompt = f"[INST] Réponds en français uniquement. Crée un rapport structuré (introduction, contenu principal, conclusion) basé uniquement sur le contenu des fichiers extraits de l’index FAISS concernant le sujet '{sujet}'. Ne génère rien hors sujet, ne mentionne pas 'segments', et n’ajoute pas d’excuses :\nContenu des fichiers : {contexte_general}\nListe des fichiers : {', '.join(self.fichiers_segments.keys())}\nRapport (en français uniquement) : [/INST]"
        reponse = generate_dynamic_response(prompt, initial_max_tokens=1000)
        self.results_text.setText(reponse)

    def ouvrir_fichier(self):
        if not self.selected_file:
            self.results_text.setText("Veuillez sélectionner un fichier à ouvrir.")
            return
        chemin_fichier = os.path.join(DOSSIER_FICHIERS, self.selected_file)
        if os.path.exists(chemin_fichier):
            try:
                subprocess.run(["xdg-open", chemin_fichier], check=True)
                self.results_text.setText(f"Fichier {self.selected_file} ouvert avec succès.")
            except subprocess.CalledProcessError:
                self.results_text.setText(f"Erreur lors de l'ouverture de {self.selected_file}.")
        else:
            self.results_text.setText(f"Le fichier {self.selected_file} n’existe pas.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MemoryOSAIWindow()
    window.show()
    sys.exit(app.exec_())
