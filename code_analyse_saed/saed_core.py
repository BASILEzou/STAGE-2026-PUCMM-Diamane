"""
Moteur de calcul cristallographique SAED Unifié.
Gère l'extraction matricielle, le traitement du signal 1D, le filtrage et la classification.
"""

import cv2
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.signal import find_peaks
import tifffile
from skimage.morphology import white_tophat, disk

# ==========================================
# BASES DE DONNÉES THÉORIQUES
# ==========================================
REFERENCE_STRUCTURES_5keV = {
    "1LG_A": { "symetrie": 6, "ring1_weak": 0.784, "ring1_strong": 0.784, "ring1_s/ring1_w": 1.0, "ring2": 0.695, "ring2/ring1_s": 0.9 },
    "2LG_AA": { "symetrie": 6, "ring1_weak": 1.505, "ring1_strong": 1.505, "ring1_s/ring1_w": 1.0, "ring2": 0.940, "ring2/ring1_s": 0.6 },
    "2LG_AB": { "symetrie": 3, "ring1_weak": 0.157, "ring1_strong": 0.698, "ring1_s/ring1_w": 4.4, "ring2": 0.940, "ring2/ring1_s": 1.4 },
    "Diamane_AB": { "symetrie": 3, "ring1_weak": 0.056, "ring1_strong": 0.362, "ring1_s/ring1_w": 6.5, "ring2": 1.03, "ring2/ring1_s": 2.8, "a1": 2.53 },
    "Diamane_AA": { "symetrie": 6, "ring1_weak": 1.364, "ring1_strong": 1.362, "ring1_s/ring1_w": 1.0, "ring2": 1.04, "ring2/ring1_s": 0.8, "a1": 2.52 },
    "Graphane": { "symetrie": 6, "ring1_weak": 0.511, "ring1_strong": 0.714, "ring1_s/ring1_w": 1.4, "ring2": 0.539, "ring2/ring1_s": 0.8, "a1": 2.53 }
}

REFERENCE_STRUCTURES_100keV = {
    "1LG_A": { "symetrie": 6, "ring1_weak": 0.784, "ring1_strong": 0.784, "ring1_s/ring1_w": 1.0, "ring2": 0.695, "ring2/ring1_s": 0.9 },
    "2LG_AA": { "symetrie": 6, "ring1_weak": 1.564, "ring1_strong": 1.564, "ring1_s/ring1_w": 1.0, "ring2": 1.367, "ring2/ring1_s": 0.9 },
    "2LG_AB": { "symetrie": 3, "ring1_weak": 0.335, "ring1_strong": 0.451, "ring1_s/ring1_w": 1.3, "ring2": 1.367, "ring2/ring1_s": 3.0 },
    "Diamane_AB": { "symetrie": 3, "ring1_weak": 0.140, "ring1_strong": 0.205, "ring1_s/ring1_w": 1.5, "ring2": 1.228, "ring2/ring1_s": 6.0, "a1": 2.53 },
    "Diamane_AA": { "symetrie": 6, "ring1_weak": 1.393, "ring1_strong": 1.393, "ring1_s/ring1_w": 1.0, "ring2": 1.24, "ring2/ring1_s": 0.9, "a1": 2.52 },
    "Graphane": { "symetrie": 6, "ring1_weak": 0.590, "ring1_strong": 0.633, "ring1_s/ring1_w": 1.1, "ring2": 0.561, "ring2/ring1_s": 0.9, "a1": 2.53 }
}

# ==========================================
# MOTEUR ALGORITHMIQUE
# ==========================================
class AnalyseurSAED:
    def __init__(self):
        self.image_brute = None
        self.image_analysee = None
        self.cx = 0.0
        self.cy = 0.0
        
        # Base de données d'acquisition manuelle
        self.data_manuel = {'R1_Strong': [], 'R1_Weak': [], 'R2': []}

    def charger_image(self, chemin):
        """Chargement en Float32 (compatible TIF et standard)."""
        if chemin.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(chemin)
        else:
            img = cv2.imread(chemin, cv2.IMREAD_ANYDEPTH)
            if img is not None and img.ndim == 3:
                img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                
        if img is None:
            raise ValueError("Fichier introuvable ou matrice vide.")
        
        self.image_brute = img.astype(np.float32)
        self.image_analysee = self.image_brute.copy()
        return self.image_brute

    def appliquer_traitement_spatial(self, x1, x2, y1, y2, mode="Crop"):
        """Opération de recadrage ou de masquage binaire."""
        if self.image_brute is None: return
        
        h, w = self.image_brute.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if mode == "Crop":
            self.image_analysee = self.image_brute[y1:y2, x1:x2].copy()
        elif mode == "Masque Noir":
            self.image_analysee = self.image_brute.copy()
            self.image_analysee[y1:y2, x1:x2] = 0

    def appliquer_tophat(self, rayon):
        """Soustraction de fond continu (Top-Hat morphology)."""
        if self.image_analysee is not None:
            self.image_analysee = white_tophat(self.image_analysee, disk(rayon))

    def extraire_intensite_spot_manuel(self, x, y, rayon):
        """Intégration d'un spot local sélectionné au clic."""
        Y, X = np.ogrid[:self.image_analysee.shape[0], :self.image_analysee.shape[1]]
        masque = (X - int(x))**2 + (Y - int(y))**2 <= rayon**2
        return np.mean(self.image_analysee[masque])

    def calculer_barycentre(self, rayon_recherche, seuil_pct):
        """Détection automatique du faisceau direct transmis."""
        hauteur, largeur = self.image_analysee.shape
        y_approx, x_approx = hauteur // 2, largeur // 2
        
        yd, yf = max(0, y_approx - rayon_recherche), min(hauteur, y_approx + rayon_recherche)
        xd, xf = max(0, x_approx - rayon_recherche), min(largeur, x_approx + rayon_recherche)
        
        roi = self.image_analysee[yd:yf, xd:xf]
        seuil = np.max(roi) * (seuil_pct / 100.0)
        matrice_seuil = np.where(roi > seuil, roi, 0)
        
        cy_roi, cx_roi = center_of_mass(matrice_seuil)
        if np.isnan(cy_roi) or np.isnan(cx_roi):
            raise ValueError("Échec du calcul du barycentre. Ajustez le seuil.")
            
        self.cy, self.cx = yd + cy_roi, xd + cx_roi
        return self.cx, self.cy

    def integrer_profil_radial(self):
        """Déploiement radial 1D."""
        y_ind, x_ind = np.indices(self.image_analysee.shape)
        dist_r = np.sqrt((x_ind - self.cx)**2 + (y_ind - self.cy)**2).astype(np.int32)
        
        somme_i = np.bincount(dist_r.ravel(), self.image_analysee.ravel())
        compte_p = np.bincount(dist_r.ravel())
        m_valide = compte_p > 0
        p_radial = np.zeros_like(somme_i, dtype=np.float64)
        p_radial[m_valide] = somme_i[m_valide] / compte_p[m_valide]
        return p_radial

    def integrer_profil_azimutal(self, r_cible, epaisseur):
        """Extraction circulaire pour détection des spots."""
        y_ind, x_ind = np.indices(self.image_analysee.shape)
        dy, dx = y_ind - self.cy, x_ind - self.cx
        r_mat = np.sqrt(dx**2 + dy**2)
        ang_mat = np.degrees(np.arctan2(dy, dx)) % 360
        bins_ang = np.arange(0, 361, 1)
        
        m_ann = (r_mat >= (r_cible - epaisseur)) & (r_mat <= (r_cible + epaisseur))
        a_iso, i_iso = ang_mat[m_ann], self.image_analysee[m_ann]
        si, _ = np.histogram(a_iso, bins=bins_ang, weights=i_iso)
        cp, _ = np.histogram(a_iso, bins=bins_ang)
        p_az = np.divide(si, cp, out=np.zeros(len(si), dtype=np.float64), where=cp!=0)
        
        lissage = np.ones(5) / 5
        return np.convolve(p_az, lissage, mode='same')

    def classifier_structure(self, tension, sym_choisie, spots_r1_s, spots_r1_w, spots_r2, methode_r2="Azimutal", i_ring2_calc=1.0, mode="Auto"):
        """Classification multi-méthodes (Auto ou Manuel)."""
        if mode == "Manuel":
            moy_forts_3 = np.mean(spots_r1_s) if len(spots_r1_s) > 0 else 1.0
            moy_faibles_3 = np.mean(spots_r1_w) if len(spots_r1_w) > 0 else 1.0
            ratio_s_3 = moy_forts_3 / moy_faibles_3 if moy_faibles_3 > 0 else 1.0
            
            tous_r1 = spots_r1_s + spots_r1_w
            moy_g_6 = np.mean(tous_r1) if len(tous_r1) > 0 else 1.0
            ratio_s_6 = 1.0
        else:
            # spots_r1_s contient l'ensemble des spots R1 détectés pour le mode Auto
            spots_t_r1 = np.sort(spots_r1_s)
            moy_forts_3 = np.mean(spots_t_r1[-3:]) if len(spots_t_r1) >= 3 else (np.mean(spots_t_r1) if len(spots_t_r1)>0 else 1.0)
            moy_faibles_3 = np.mean(spots_t_r1[:3]) if len(spots_t_r1) >= 3 else (np.mean(spots_t_r1) if len(spots_t_r1)>0 else 1.0)
            ratio_s_3 = moy_forts_3 / moy_faibles_3 if moy_faibles_3 > 0 else 1.0
            
            moy_g_6 = np.mean(spots_t_r1) if len(spots_t_r1) > 0 else 1.0
            ratio_s_6 = 1.0

        intensite_r2 = i_ring2_calc if methode_r2 == "Radial" else (np.mean(spots_r2) if len(spots_r2) > 0 else 1.0)
        db_active = REFERENCE_STRUCTURES_5keV if tension == 5 else REFERENCE_STRUCTURES_100keV
        
        res_match_sym, res_autre_sym = [], []
        
        for nom, ref in db_active.items():
            sym_theorique = ref.get("symetrie", 6)
            ratio_s_exp = ratio_s_3 if sym_theorique == 3 else ratio_s_6
            moy_forts_exp = moy_forts_3 if sym_theorique == 3 else moy_g_6
            
            ratio_g_exp = intensite_r2 / moy_forts_exp if moy_forts_exp > 0 else 0
            err = (ratio_g_exp - ref["ring2/ring1_s"])**2 + (ratio_s_exp - ref["ring1_s/ring1_w"])**2
            
            resultat = (nom, err, ratio_s_exp, ratio_g_exp)
            if sym_theorique == sym_choisie:
                res_match_sym.append(resultat)
            else:
                res_autre_sym.append(resultat)
                
        res_match_sym.sort(key=lambda x: x[1])
        res_autre_sym.sort(key=lambda x: x[1])
        
        return res_match_sym, res_autre_sym, ratio_s_3, ratio_s_6, intensite_r2, moy_forts_3, moy_faibles_3, moy_g_6