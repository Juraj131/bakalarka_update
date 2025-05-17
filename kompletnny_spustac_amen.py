import time
# Importujeme moduly, ktoré obsahujú naše hlavné tréningové funkcie
# Predpokladáme, že tieto moduly sú v rovnakom adresári
import u_net_2_training_calss as klasifikacny_modul
import u_net_2_training_regresion as regresny_modul

def hlavny_spustac_vsetkych_treningov():
    """
    Hlavná funkcia, ktorá postupne spúšťa všetky definované tréningové sekvencie.
    """
    print("=" * 70)
    print("KOMPLETNÝ SPÚŠŤAČ: ŠTART VŠETKÝCH TRÉNINGOVÝCH SEKVENCIÍ")
    print("=" * 70)
    celkovy_start_cas = time.time()

    # ----------------------------------------------------
    # 1. Spustenie všetkých KLASIFIKAČNÝCH tréningov
    # ----------------------------------------------------
    print("\n" + "-" * 60)
    print("Spúšťam KLASIFIKAČNÉ tréningy...")
    print("-" * 60)
    start_cas_klasifikacia = time.time()
    try:
        # Volanie funkcie z modulu u_net_2_training_calss.py
        klasifikacny_modul.spusti_vsetky_klasifikacne_treningy()
        trvanie_klasifikacia = time.time() - start_cas_klasifikacia
        print("-" * 60)
        print(f"KLASIFIKAČNÉ tréningy dokončené úspešne. Trvanie: {trvanie_klasifikacia:.2f} s")
        print("-" * 60)
    except Exception as e:
        trvanie_klasifikacia = time.time() - start_cas_klasifikacia
        print("-" * 60)
        print(f"!!! CHYBA počas KLASIFIKAČNÝCH tréningov: {e} !!!")
        print(f"Trvanie pred chybou: {trvanie_klasifikacia:.2f} s")
        print("-" * 60)
        # Môžete sa rozhodnúť, či pokračovať alebo ukončiť skript v prípade chyby
        # napr. return

    # ----------------------------------------------------
    # 2. Spustenie všetkých REGRESNÝCH tréningov
    # ----------------------------------------------------
    print("\n" + "-" * 60)
    print("Spúšťam REGRESNÉ tréningy...")
    print("-" * 60)
    start_cas_regresia = time.time()
    try:
        # Volanie funkcie z modulu u_net_training_regresion.py
        regresny_modul.spusti_vsetky_regresne_treningy()
        trvanie_regresia = time.time() - start_cas_regresia
        print("-" * 60)
        print(f"REGRESNÉ tréningy dokončené úspešne. Trvanie: {trvanie_regresia:.2f} s")
        print("-" * 60)
    except Exception as e:
        trvanie_regresia = time.time() - start_cas_regresia
        print("-" * 60)
        print(f"!!! CHYBA počas REGRESNÝCH tréningov: {e} !!!")
        print(f"Trvanie pred chybou: {trvanie_regresia:.2f} s")
        print("-" * 60)
        # Môžete sa rozhodnúť, či pokračovať alebo ukončiť skript v prípade chyby
        # napr. return

    celkove_trvanie = time.time() - celkovy_start_cas
    print("\n" + "=" * 70)
    print(f"KOMPLETNÝ SPÚŠŤAČ: VŠETKY TRÉNINGOVÉ SEKVENCIE DOKONČENÉ.")
    print(f"Celkové trvanie všetkých operácií: {celkove_trvanie:.2f} s")
    print("=" * 70)

if __name__ == '__main__':
    # Tento blok sa vykoná, keď spustíte skript kompletnny_spustac_amen.py priamo
    hlavny_spustac_vsetkych_treningov()