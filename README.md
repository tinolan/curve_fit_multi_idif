# Kinetic Model 2TC Curve Fit für PET-Daten

## Projektbeschreibung
Dieses Repository enthält ein Python-Skript zur Analyse von PET-Daten (Positronen-Emissions-Tomographie) mittels eines Zweikompartimentmodells (2TC). Die Hauptaufgabe des Skripts besteht darin, Kinetikparameter für verschiedene Regionen der Lunge zu berechnen und zu speichern.

## Voraussetzungen
Bevor das Skript ausgeführt werden kann, müssen einige Abhängigkeiten installiert sein.

### Erforderliche Python-Bibliotheken:
- `numpy`
- `pandas`
- `torch`
- `SimpleITK`
- `matplotlib`
- `scipy`
- `natsort`

Diese können mit folgendem Befehl installiert werden:
```bash
pip install numpy pandas torch SimpleITK matplotlib scipy natsort
```

## Verzeichnisstruktur
```
/
├── utils/
│   ├── utils_torch.py  # Enthält Funktionen für Torch-Interpolation und Faltung
│   ├── set_root_paths.py  # Enthält Variablen für Datenpfade
├── main.py  # Hauptskript zur PET-Datenverarbeitung
├── README.md  # Diese Datei
```

## Funktionsbeschreibung

### `reduce_to_600(values)`
Reduziert eine gegebene Zeitreihe auf 600 Werte, indem es den maximalen Wert in definierten Intervallen extrahiert.

### `KineticModel_2TC_curve_fit`
Eine Klasse zur Modellierung von Kinetikparametern in PET-Daten. Enthält Methoden:
- `read_idif(sample_time, t)`: Liest und interpoliert IDIF-Werte aus CSV-Dateien.
- `PET_2TC_KM(t, k1, k2, k3, Vb, alpha, beta)`: Berechnet die PET-Daten mittels eines Zweikompartimentmodells.
- `PET_normal(t, k1, k2, k3, Vb)`: Berechnet PET-Daten mit einem vereinfachten Modell.

### `process_pet_lung(patient)`
Führt die Modellierung für die Lunge durch, einschließlich der Nutzung von Aorta- und Pulmonalarterien-IDIF.

### `process_pet_lung_normal(patient)`
Verarbeitet die Lunge nur mit Aorta-IDIF.

## Datenspeicherung
Die Parameter werden als `.npz`-Dateien gespeichert:

## Fehlerbehandlung
Falls bei der Anpassung der Parameter Fehler auftreten, wird eine Fehlermeldung mit den betroffenen Voxel-Koordinaten ausgegeben.

## Lizenz
MIT License

