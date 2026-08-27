"""
Point d'entrée exécutable de l'application SAED.
"""

import tkinter as tk
from saed_gui import InterfaceSAED

def main():
    racine = tk.Tk()
    app = InterfaceSAED(racine)
    racine.mainloop()

if __name__ == "__main__":
    main()