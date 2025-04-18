from moviepy import VideoFileClip, CompositeVideoClip, ImageClip
import numpy as np
import random

# Charger la vidéo d'origine (20 fps = 20s)
clip = VideoFileClip("/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/animations/Média1.mp4")

# Charger les icônes
icon_stop = ImageClip("/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/animations/icons8-arrêter-60.png").resized(height=300)
icon_go = ImageClip("/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/animations/icons8-ok-main-skin-type-2-48.png").resized(height=300)

# Générer les timestamps (1 par seconde, donc une icône par frame)
timestamps = np.arange(0, clip.duration, 1.0)  # 20 timestamps si durée = 20s

# Choix aléatoire ou défini des icônes par frame
icons = [random.choice([icon_stop, icon_go]) for _ in timestamps]

# Créer les overlays : chaque icône dure exactement 1s
overlays = [
    icon.with_position((2000, 2000)).with_start(t).with_duration(1.0)
    for t, icon in zip(timestamps, icons)
]

# Fusionner vidéo + overlays
final = CompositeVideoClip([clip] + overlays)

# Exporter
final.write_videofile("/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/animations/output_avec_icones_par_frame.mp4", codec="libx264", fps=1)
