"""
Interface graphique Tkinter pour l'analyse SAED Unifiée.
Gère les modes Manuel (Clics) et Automatique (Profils), le rendu colorimétrique (CMap), les légendes et l'exportation globale.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks

from saed_core import AnalyseurSAED

class InterfaceSAED:
    def __init__(self, master):
        self.master = master
        self.moteur = AnalyseurSAED()
        
        master.title("Analyseur SAED Unifié (Manuel & Profils)")
        master.geometry("1600x950")
        
        self.chemin_fichier = tk.StringVar()
        self.dernier_mode_utilise = "Aucun"
        
        # Tampons de données pour l'exportation
        self.log_manuel = ""
        self.log_auto = ""
        
        # Variables Pré-traitement
        self.var_x1 = tk.IntVar(value=0)
        self.var_x2 = tk.IntVar(value=0)
        self.var_y1 = tk.IntVar(value=0)
        self.var_y2 = tk.IntVar(value=0)
        self.var_type_masque = tk.StringVar(value="Crop")
        self.var_tophat = tk.BooleanVar(value=False)
        self.var_rayon_tophat = tk.IntVar(value=50)
        
        # Variables Physiques
        self.var_tension = tk.IntVar(value=5)
        self.var_symetrie = tk.IntVar(value=6)
        
        # Variables Mode Manuel
        self.var_rayon_spot = tk.IntVar(value=15)
        self.var_cible_manuelle = tk.StringVar(value="R1_Strong")
        self.cercles_patches = []
        
        # Variables Mode Auto
        self.var_manuel_centre = tk.BooleanVar(value=False)
        self.var_cx = tk.DoubleVar(value=500.0)
        self.var_cy = tk.DoubleVar(value=500.0)
        self.var_rayon_recherche = tk.IntVar(value=50)
        self.var_seuil = tk.DoubleVar(value=90.0)
        self.var_anneaux_manuels = tk.BooleanVar(value=False)
        self.var_rmin = tk.IntVar(value=50)
        self.var_rmax = tk.IntVar(value=200)
        self.var_r1 = tk.IntVar(value=100)
        self.var_r2 = tk.IntVar(value=200)
        self.var_prom_rad = tk.DoubleVar(value=2.0)
        self.var_dist_rad = tk.IntVar(value=10)
        self.var_epaisseur = tk.IntVar(value=1)
        self.var_methode_ratio = tk.StringVar(value="Azimutal")
        self.var_prom_az_r1 = tk.DoubleVar(value=5.0)
        self.var_dist_az_r1 = tk.IntVar(value=10)
        self.var_prom_az_r2 = tk.DoubleVar(value=5.0)
        self.var_dist_az_r2 = tk.IntVar(value=10)

        self._creer_widgets()

    def _creer_widgets(self):
        panneau_gauche = tk.Frame(self.master, width=450, padx=5, pady=5)
        panneau_gauche.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        
        panneau_droit = tk.Frame(self.master, padx=5, pady=5)
        panneau_droit.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. CHARGEMENT
        frm_fichier = tk.LabelFrame(panneau_gauche, text="1. Fichier Image")
        frm_fichier.pack(fill=tk.X, pady=2)
        tk.Button(frm_fichier, text="Charger", command=self._choisir_fichier).grid(row=0, column=0, padx=5, pady=2)
        tk.Label(frm_fichier, textvariable=self.chemin_fichier, wraplength=300).grid(row=0, column=1, padx=5)

        # 2. PRE-TRAITEMENTS
        frm_param = tk.LabelFrame(panneau_gauche, text="2. Redimensionnement & Filtrage")
        frm_param.pack(fill=tk.X, pady=2)
        tk.Label(frm_param, text="x1/x2:").grid(row=0, column=0, sticky="e")
        tk.Entry(frm_param, textvariable=self.var_x1, width=5).grid(row=0, column=1)
        tk.Entry(frm_param, textvariable=self.var_x2, width=5).grid(row=0, column=2)
        tk.Label(frm_param, text="y1/y2:").grid(row=1, column=0, sticky="e")
        tk.Entry(frm_param, textvariable=self.var_y1, width=5).grid(row=1, column=1)
        tk.Entry(frm_param, textvariable=self.var_y2, width=5).grid(row=1, column=2)
        ttk.Combobox(frm_param, textvariable=self.var_type_masque, values=["Crop", "Masque Noir"], width=10).grid(row=0, column=3, rowspan=2, padx=5)
        tk.Checkbutton(frm_param, text="Top-Hat", variable=self.var_tophat).grid(row=2, column=0, columnspan=2, sticky="w")
        tk.Label(frm_param, text="Rayon (px):").grid(row=2, column=2, sticky="e")
        tk.Entry(frm_param, textvariable=self.var_rayon_tophat, width=5).grid(row=2, column=3)
        tk.Button(frm_param, text="Appliquer Pré-traitements", command=self.appliquer_pre_traitements).grid(row=3, column=0, columnspan=4, pady=5)

        # 3. PHYSIQUE
        frm_phys = tk.LabelFrame(panneau_gauche, text="3. Paramètres Physiques")
        frm_phys.pack(fill=tk.X, pady=2)
        tk.Label(frm_phys, text="Tension:").grid(row=0, column=0)
        tk.Radiobutton(frm_phys, text="5 keV", variable=self.var_tension, value=5).grid(row=0, column=1)
        tk.Radiobutton(frm_phys, text="100 keV", variable=self.var_tension, value=100).grid(row=0, column=2)
        tk.Label(frm_phys, text="Symétrie:").grid(row=1, column=0)
        tk.Radiobutton(frm_phys, text="Ordre 3", variable=self.var_symetrie, value=3).grid(row=1, column=1)
        tk.Radiobutton(frm_phys, text="Ordre 6", variable=self.var_symetrie, value=6).grid(row=1, column=2)

        # 4. METHODES
        self.nb_methodes = ttk.Notebook(panneau_gauche)
        self.nb_methodes.pack(fill=tk.X, pady=5)

        # -- MANUEL --
        tab_manuel = tk.Frame(self.nb_methodes)
        self.nb_methodes.add(tab_manuel, text="Mode Manuel (Clics)")
        tk.Label(tab_manuel, text="Rayon du spot (px):").grid(row=0, column=0, pady=5)
        tk.Entry(tab_manuel, textvariable=self.var_rayon_spot, width=5).grid(row=0, column=1)
        tk.Radiobutton(tab_manuel, text="Ring 1 : Spots Forts", variable=self.var_cible_manuelle, value="R1_Strong").grid(row=1, column=0, columnspan=2, sticky="w")
        tk.Radiobutton(tab_manuel, text="Ring 1 : Spots Faibles", variable=self.var_cible_manuelle, value="R1_Weak").grid(row=2, column=0, columnspan=2, sticky="w")
        tk.Radiobutton(tab_manuel, text="Ring 2", variable=self.var_cible_manuelle, value="R2").grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Button(tab_manuel, text="Évaluer Spots Manuels", command=self.evaluer_mode_manuel, bg="#ffcc99").grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

        # -- AUTO --
        tab_auto = tk.Frame(self.nb_methodes)
        self.nb_methodes.add(tab_auto, text="Mode Auto (Profils)")
        
        f_c = tk.LabelFrame(tab_auto, text="Centre & Rayons (px)")
        f_c.pack(fill=tk.X)
        tk.Checkbutton(f_c, text="Forcer XY", variable=self.var_manuel_centre).grid(row=0, column=0)
        tk.Entry(f_c, textvariable=self.var_cx, width=4).grid(row=0, column=1)
        tk.Entry(f_c, textvariable=self.var_cy, width=4).grid(row=0, column=2)
        tk.Label(f_c, text="Rech/Seuil:").grid(row=1, column=0)
        tk.Entry(f_c, textvariable=self.var_rayon_recherche, width=4).grid(row=1, column=1)
        tk.Entry(f_c, textvariable=self.var_seuil, width=4).grid(row=1, column=2)
        tk.Label(f_c, text="Rmin/Rmax:").grid(row=2, column=0)
        tk.Entry(f_c, textvariable=self.var_rmin, width=4).grid(row=2, column=1)
        tk.Entry(f_c, textvariable=self.var_rmax, width=4).grid(row=2, column=2)

        f_a = tk.LabelFrame(tab_auto, text="Détection & Azimutal")
        f_a.pack(fill=tk.X)
        tk.Checkbutton(f_a, text="Forcer R1/R2", variable=self.var_anneaux_manuels).grid(row=0, column=0)
        tk.Entry(f_a, textvariable=self.var_r1, width=4).grid(row=0, column=1)
        tk.Entry(f_a, textvariable=self.var_r2, width=4).grid(row=0, column=2)
        tk.Label(f_a, text="Tolérance (± ep):").grid(row=1, column=0)
        tk.Entry(f_a, textvariable=self.var_epaisseur, width=4).grid(row=1, column=1)
        
        tk.Label(f_a, text="Prom/Dist Rad.:").grid(row=2, column=0)
        tk.Entry(f_a, textvariable=self.var_prom_rad, width=4).grid(row=2, column=1)
        tk.Entry(f_a, textvariable=self.var_dist_rad, width=4).grid(row=2, column=2)

        tk.Label(f_a, text="Promin. Az R1/R2:").grid(row=3, column=0)
        tk.Entry(f_a, textvariable=self.var_prom_az_r1, width=4).grid(row=3, column=1)
        tk.Entry(f_a, textvariable=self.var_prom_az_r2, width=4).grid(row=3, column=2)
        tk.Button(tab_auto, text="Lancer Analyse Auto", command=self.analyser_mode_auto, bg="#99ff99").pack(fill=tk.X, pady=5)

        # 5. RESULTATS
        frm_res = tk.LabelFrame(panneau_gauche, text="5. Résultats")
        frm_res.pack(fill=tk.BOTH, expand=True)
        frm_btn = tk.Frame(frm_res)
        frm_btn.pack(fill=tk.X, pady=2)
        tk.Button(frm_btn, text="Reset Data", command=self.reset_donnees).pack(side=tk.LEFT, padx=2)
        tk.Button(frm_btn, text="Exporter Rapport Global", command=self.exporter_rapport, bg="lightblue").pack(side=tk.RIGHT, padx=2)
        
        self.txt_resultats = tk.Text(frm_res, height=12)
        self.txt_resultats.pack(fill=tk.BOTH, expand=True)

        # FIGURES (DROITE)
        self.nb_visu = ttk.Notebook(panneau_droit)
        self.nb_visu.pack(fill=tk.BOTH, expand=True)
        
        # Onglet Image (Canvas instancié en premier, pack de la toolbar, puis pack du canvas)
        tab_img = tk.Frame(self.nb_visu)
        self.nb_visu.add(tab_img, text="Cliché Interactif")
        self.fig_img = plt.figure(figsize=(8, 8))
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=tab_img)
        self.toolbar_img = NavigationToolbar2Tk(self.canvas_img, tab_img)
        self.toolbar_img.update()
        self.canvas_img.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_img.mpl_connect('button_press_event', self.clic_plot)

        # Onglet Profils (Canvas instancié en premier, pack de la toolbar, puis pack du canvas)
        tab_prof = tk.Frame(self.nb_visu)
        self.nb_visu.add(tab_prof, text="Profils 1D (Auto)")
        self.fig_prof = plt.figure(figsize=(8, 8))
        self.gs = gridspec.GridSpec(2, 1)
        self.ax_rad = self.fig_prof.add_subplot(self.gs[0, 0])
        self.ax_az = self.fig_prof.add_subplot(self.gs[1, 0])
        self.canvas_prof = FigureCanvasTkAgg(self.fig_prof, master=tab_prof)
        self.toolbar_prof = NavigationToolbar2Tk(self.canvas_prof, tab_prof)
        self.toolbar_prof.update()
        self.canvas_prof.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._reset_axes()

    def _reset_axes(self):
        self.fig_img.clf()
        self.ax_img = self.fig_img.add_subplot(111)
        self.ax_img.set_axis_off()
        self.ax_rad.clear()
        self.ax_rad.set_title("Profil Radial 1D")
        self.ax_az.clear()
        self.ax_az.set_title("Profil Azimutal R1 et R2")
        self.fig_img.tight_layout()
        self.fig_prof.tight_layout()

    def _choisir_fichier(self):
        fichier = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff")])
        if fichier:
            self.chemin_fichier.set(fichier)
            img = self.moteur.charger_image(fichier)
            h, w = img.shape
            self.var_x1.set(0); self.var_x2.set(w)
            self.var_y1.set(0); self.var_y2.set(h)
            self.appliquer_pre_traitements()

    def _ecrire_log(self, message, mode):
        if mode == "Manuel":
            self.log_manuel += message + "\n"
        elif mode == "Auto":
            self.log_auto += message + "\n"
        self.txt_resultats.insert(tk.END, message + "\n")
        self.txt_resultats.see(tk.END)

    def appliquer_pre_traitements(self):
        if self.moteur.image_brute is None: return
        self.moteur.appliquer_traitement_spatial(
            self.var_x1.get(), self.var_x2.get(), 
            self.var_y1.get(), self.var_y2.get(), mode=self.var_type_masque.get()
        )
        if self.var_tophat.get():
            self.moteur.appliquer_tophat(self.var_rayon_tophat.get())
        self.afficher_image()

    def afficher_image(self):
        self.fig_img.clf()
        self.ax_img = self.fig_img.add_subplot(111)
        self.ax_img.set_axis_off()
        
        if self.moteur.image_analysee is not None:
            self.ax_img.set_axis_on()
            im = self.ax_img.imshow(self.moteur.image_analysee, cmap='gray')
            
            divider = make_axes_locatable(self.ax_img)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            self.fig_img.colorbar(im, cax=cax, label="Intensité absolue")
            
        if self.dernier_mode_utilise == "Auto" or self.var_manuel_centre.get():
            self.ax_img.plot(self.moteur.cx, self.moteur.cy, 'r+', markersize=12, markeredgewidth=2, label="Centre Faisceau")
            
        for c in self.cercles_patches:
            self.ax_img.add_patch(patches.Circle((c['x'], c['y']), c['r'], edgecolor=c['color'], facecolor='none', lw=1.5))
            self.ax_img.text(c['x'] + c['r'], c['y'], f"{c['val']:.1f}", color=c['color'], fontsize=8)

        self.fig_img.tight_layout()
        self.canvas_img.draw()

    # ==========================
    # LOGIQUE MODE MANUEL
    # ==========================
    def clic_plot(self, event):
        onglet_actif = self.nb_methodes.tab(self.nb_methodes.select(), "text")
        if onglet_actif != "Mode Manuel (Clics)": return
        if self.toolbar_img.mode != "" or event.button != 1 or event.inaxes != self.ax_img or self.moteur.image_analysee is None: return
            
        xc, yc = float(event.xdata), float(event.ydata)
        r = self.var_rayon_spot.get()
        
        moyenne = self.moteur.extraire_intensite_spot_manuel(xc, yc, r)
        cible = self.var_cible_manuelle.get()
        
        self.moteur.data_manuel[cible].append(moyenne)
        couleur = {'R1_Strong': 'red', 'R1_Weak': 'orange', 'R2': 'cyan'}[cible]
        
        self.cercles_patches.append({'x': xc, 'y': yc, 'r': r, 'val': moyenne, 'color': couleur})
        self.dernier_mode_utilise = "Manuel"
        self.afficher_image()

    def evaluer_mode_manuel(self):
        self.txt_resultats.delete(1.0, tk.END)
        self.log_manuel = ""
        self.dernier_mode_utilise = "Manuel"
        
        spots_r1_s = self.moteur.data_manuel['R1_Strong']
        spots_r1_w = self.moteur.data_manuel['R1_Weak']
        spots_r2 = self.moteur.data_manuel['R2']
        
        self._ecrire_log(f"--- ANALYSE MANUELLE ---", "Manuel")
        self._ecrire_log(f"Spots (R1 Forts) : {len(spots_r1_s)} | (R1 Faibles) : {len(spots_r1_w)} | (R2) : {len(spots_r2)}", "Manuel")
        
        if len(spots_r1_s) + len(spots_r1_w) == 0:
            self._ecrire_log("Veuillez acquérir des spots R1.", "Manuel")
            return

        res_match, res_autre, r_s3, r_s6, int_r2, m_f3, m_w3, m_g6 = self.moteur.classifier_structure(
            self.var_tension.get(), self.var_symetrie.get(), spots_r1_s, spots_r1_w, spots_r2, mode="Manuel"
        )
        self._afficher_verdict(res_match, res_autre, r_s3, r_s6, int_r2, m_f3, m_w3, m_g6, "Manuel")

    # ==========================
    # LOGIQUE MODE AUTOMATIQUE
    # ==========================
    def analyser_mode_auto(self):
        if self.moteur.image_analysee is None: return
        self.txt_resultats.delete(1.0, tk.END)
        self.log_auto = ""
        self.dernier_mode_utilise = "Auto"
        
        # 1. Barycentre
        if self.var_manuel_centre.get():
            self.moteur.cx, self.moteur.cy = self.var_cx.get(), self.var_cy.get()
        else:
            try:
                cx, cy = self.moteur.calculer_barycentre(self.var_rayon_recherche.get(), self.var_seuil.get())
                self.var_cx.set(round(cx, 2)); self.var_cy.set(round(cy, 2))
            except Exception as e:
                messagebox.showerror("Erreur", str(e))
                return

        # 2. Radial
        r_min, r_max = self.var_rmin.get(), self.var_rmax.get()
        p_radial = self.moteur.integrer_profil_radial()
        r_max_eff = min(r_max, len(p_radial))

        if self.var_anneaux_manuels.get():
            r_ring1, r_ring2 = self.var_r1.get(), self.var_r2.get()
            r_pics = np.array([r_ring1, r_ring2])
            i_ring1_calc = p_radial[r_ring1] - np.min(p_radial[max(0, r_ring1-5):min(len(p_radial), r_ring1+5)])
            i_ring2_calc = p_radial[r_ring2] - np.min(p_radial[max(0, r_ring2-5):min(len(p_radial), r_ring2+5)])
            i_pics_disp = np.array([p_radial[r_ring1], p_radial[r_ring2]])
        else:
            idx_pics, prop_rad = find_peaks(p_radial[r_min:r_max_eff], prominence=self.var_prom_rad.get(), distance=self.var_dist_rad.get())
            if len(idx_pics) < 2:
                messagebox.showerror("Erreur Auto", "Impossible de détecter 2 anneaux continus. Baissez la prominence radiale.")
                return
            r_pics = idx_pics + r_min
            r_ring1, r_ring2 = r_pics[0], r_pics[1]
            i_ring1_calc, i_ring2_calc = prop_rad["prominences"][0], prop_rad["prominences"][1]
            i_pics_disp = p_radial[r_pics]
            self.var_r1.set(r_ring1); self.var_r2.set(r_ring2)

        # 3. Azimutal
        ep = self.var_epaisseur.get()
        p_az_r1 = self.moteur.integrer_profil_azimutal(r_ring1, ep)
        p_az_r2 = self.moteur.integrer_profil_azimutal(r_ring2, ep)

        idx_s1, prop_s1 = find_peaks(p_az_r1, distance=self.var_dist_az_r1.get(), prominence=self.var_prom_az_r1.get())
        idx_s2, prop_s2 = find_peaks(p_az_r2, distance=self.var_dist_az_r2.get(), prominence=self.var_prom_az_r2.get())

        self.afficher_image()
        self.ax_img.add_patch(plt.Circle((self.moteur.cx, self.moteur.cy), r_ring1, color='green', fill=False, linestyle='-', linewidth=1.5, label="Ring 1"))
        self.ax_img.add_patch(plt.Circle((self.moteur.cx, self.moteur.cy), r_ring1 + ep, color='green', fill=False, linestyle=':', linewidth=1, label="Tolérance R1"))
        self.ax_img.add_patch(plt.Circle((self.moteur.cx, self.moteur.cy), r_ring1 - ep, color='green', fill=False, linestyle=':', linewidth=1))

        self.ax_img.add_patch(plt.Circle((self.moteur.cx, self.moteur.cy), r_ring2, color='orange', fill=False, linestyle='-', linewidth=1.5, label="Ring 2"))
        self.ax_img.add_patch(plt.Circle((self.moteur.cx, self.moteur.cy), r_ring2 + ep, color='orange', fill=False, linestyle=':', linewidth=1, label="Tolérance R2"))
        self.ax_img.add_patch(plt.Circle((self.moteur.cx, self.moteur.cy), r_ring2 - ep, color='orange', fill=False, linestyle=':', linewidth=1))
        
        self.ax_img.legend(loc="upper right", fontsize='small')
        self.canvas_img.draw()

        self._ecrire_log(f"--- ANALYSE AUTOMATIQUE ---", "Auto")
        self._ecrire_log(f"Spots R1 : {len(idx_s1)} | Spots R2 : {len(idx_s2)}", "Auto")

        self.ax_rad.clear(); self.ax_az.clear()
        self.ax_rad.set_title("Profil Radial 1D"); self.ax_az.set_title("Profil Azimutal R1 et R2")
        self.ax_rad.plot(p_radial, 'b'); self.ax_rad.plot(r_pics, i_pics_disp, 'rx', markersize=10)
        self.ax_rad.axvline(x=r_ring1, color='green', linestyle='--'); self.ax_rad.axvline(x=r_ring2, color='orange', linestyle='--')
        self.ax_rad.set_xlim(0, r_max_eff + 50)
        
        bins_ang = np.arange(0, 360, 1)
        self.ax_az.plot(bins_ang, p_az_r1, 'g', label="Ring 1"); self.ax_az.plot(bins_ang, p_az_r2, 'orange', label="Ring 2")
        if len(idx_s1) > 0: self.ax_az.plot(idx_s1, p_az_r1[idx_s1], 'rx', markersize=8)
        if len(idx_s2) > 0: self.ax_az.plot(idx_s2, p_az_r2[idx_s2], 'rx', markersize=8)
        self.ax_az.set_xlim(0, 360); self.ax_az.legend(fontsize='small')
        self.fig_prof.tight_layout()
        self.canvas_prof.draw()
        
        self.nb_visu.select(1)

        # 4. Classification
        
        res_match, res_autre, r_s3, r_s6, int_r2, m_f3, m_w3, m_g6 = self.moteur.classifier_structure(
        self.var_tension.get(), self.var_symetrie.get(), prop_s1["prominences"], [], prop_s2["prominences"], 
        methode_r2=self.var_methode_ratio.get(), i_ring2_calc=i_ring2_calc, mode="Auto"
        )
        self._afficher_verdict(res_match, res_autre, r_s3, r_s6, int_r2, m_f3, m_w3, m_g6, "Auto")

    def _afficher_verdict(self, res_match, res_autre, r_s3, r_s6, int_r2, m_f3, m_w3, m_g6, mode):
        sym = self.var_symetrie.get()
        if sym == 3:
            self._ecrire_log(f"I_Fortes (R1) : {m_f3:.2f} | I_Faibles (R1) : {m_w3:.2f}", mode)
            self._ecrire_log(f"Ratios : R1s/R1w = {r_s3:.2f} | R2/R1s = {(int_r2/m_f3 if m_f3>0 else 0):.2f}", mode)
        else:
            self._ecrire_log(f"I_Moyenne (R1) : {m_g6:.2f}", mode)
            self._ecrire_log(f"Ratios : R1s/R1w = {r_s6:.2f} | R2/R1s = {(int_r2/m_g6 if m_g6>0 else 0):.2f}", mode)
            
        self._ecrire_log(f"\n--- VERDICT (Symétrie {sym}) ---", mode)
        if not res_match:
            self._ecrire_log("Aucune correspondance.", mode)
        else:
            for rang, (nom, err, r_s, r_g) in enumerate(res_match[:3], 1):
                self._ecrire_log(f"{rang}. {nom} (Err: {err:.4f})", mode)
                
        self._ecrire_log("\n--- AUTRES SYMÉTRIES ---", mode)
        for i, (nom, err, r_s, r_g) in enumerate(res_autre[:2], 1):
             self._ecrire_log(f" - {nom} (Err: {err:.4f})", mode)

    # ==========================
    # EXPORTATION
    # ==========================
    def reset_donnees(self):
        self.moteur.data_manuel = {'R1_Strong': [], 'R1_Weak': [], 'R2': []}
        self.cercles_patches = []
        self.txt_resultats.delete(1.0, tk.END)
        self.log_manuel = ""
        self.log_auto = ""
        self.dernier_mode_utilise = "Aucun"
        self._reset_axes()
        self.afficher_image()

    def exporter_rapport(self):
        if not self.log_manuel and not self.log_auto:
            messagebox.showwarning("Attention", "Aucune donnée à exporter.")
            return
            
        chemin = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Texte", "*.txt")])
        if not chemin: return
        
        try:
            with open(chemin, 'w', encoding='utf-8') as f:
                f.write("=== RAPPORT GLOBAL D'ANALYSE SAED ===\n\n")
                f.write(f"Tension : {self.var_tension.get()} keV\n")
                f.write(f"Symétrie observée : Ordre {self.var_symetrie.get()}\n\n")
                
                if self.log_manuel:
                    f.write("========== MODE MANUEL ==========\n")
                    f.write("--- DONNÉES BRUTES ---\n")
                    for cat in ['R1_Strong', 'R1_Weak', 'R2']:
                        f.write(f"[{cat}]\n")
                        for i, intensite in enumerate(self.moteur.data_manuel[cat]):
                            f.write(f"Spot {i+1} : I = {intensite:.2f}\n")
                    f.write("\n--- RESULTATS ---\n")
                    f.write(self.log_manuel)
                    f.write("\n\n")
                    
                if self.log_auto:
                    f.write("======== MODE AUTOMATIQUE ========\n")
                    f.write("--- RESULTATS ---\n")
                    f.write(self.log_auto)
                    f.write("\n")
                    
            messagebox.showinfo("Export", "Rapport généré avec succès.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec d'écriture : {str(e)}")