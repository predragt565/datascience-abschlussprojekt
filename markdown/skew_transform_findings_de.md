## 🛏️ Eurostat Touristische Übernachtungen 2012–2025 (EU10) Trendanalyse

### Analytische Befunde – die Feature-Schiefe

### 🔎 Ergebnisse der vollständigen Schiefe-Analyse

1. **Vor der Transformation (Rohdaten):**
   * Die Schiefe ist durchgängig in vielen Merkmalen sehr hoch.  
   * Mehrere Merkmale sind **extrem rechtsschief**, mit Werten deutlich über +10, und das extremste Merkmal (`pch_sm`) erreicht über **+42**.  
   * Insgesamt überschreiten **10 von 15 numerischen Merkmalen** die Schwelle von |2| und sind damit für direktes ML-Training hochproblematisch.  
   * Dies bestätigt, dass stark ausgeprägte Verteilungsschwänze ein **systemisches Datenproblem** darstellen und nicht nur wenige Spalten betreffen.

2. **Nach log1p-Transformation:**
   * Bei strikt positiven Merkmalen konnte die Schiefe deutlich reduziert werden.  
   * Viele zuvor extreme Merkmale (z. B. Lag-Features) kippten in eine moderate **Linksschiefe** (ca. -1.7 bis -2.6).  
   * Nur **4 Merkmale** liegen nach log1p weiterhin außerhalb des akzeptablen Bereichs (|Schiefe| ≥ 2).  
   * Für einige Merkmale mit negativen Werten (z. B. prozentuale Veränderungen) konnte `log1p` nicht angewendet werden. Diese Spalten bleiben in der aktuellen Transformation unbehandelt und sollten stattdessen mit **Yeo–Johnson** in einem separaten Schritt transformiert werden, um eine konsistente Abdeckung zu erreichen.

---

### ⚖️ Interpretation

* **Rohdaten**: Aufgrund starker Schiefe ungeeignet für lineare oder distanzbasierte ML-Modelle.  
* **Log1p-transformierte Daten**: Deutlich verbessert. Die meisten Merkmale sind nun näher an symmetrischen Verteilungen, ein Teil bleibt jedoch unbehandelt (wegen negativer Werte) oder weiterhin moderat schief.  
* **Überkompensation**: Wie bei den Lag-Merkmalen sichtbar, wird starke Rechtsschiefe nach log1p oft in moderate Linksschiefe überführt. Dies ist akzeptabel und verbessert die Modellstabilität insgesamt.

---

### ⚠️ Qualitätsaspekte des Datensatzes

* **Schiefe ist systemisch**: Nahezu alle numerischen Merkmale sind betroffen.  
* **Transformationsabdeckung**: Merkmale mit negativen Werten (z. B. prozentuale Veränderungen) wurden von log1p ausgeschlossen und zeigen weiterhin sehr hohe Schiefe. Diese erfordern eine alternative Transformation wie **Yeo–Johnson**.  
* **Verbleibende Schiefe**: Einige Merkmale liegen auch nach log1p außerhalb des Bereichs von |2|. Diese können zusätzlich von Yeo–Johnson oder Winsorisieren profitieren.  
* **Ausreißer**: Noch nicht direkt betrachtet. Eine Ausreißer-Analyse (IQR, Z-Score, IsolationForest) ergänzt die Schiefe-Korrektur.

---

### 📝 Empfehlungen

1. **log1p-Transformation für alle strikt positiven Merkmale anwenden.**
   * Hat sich bereits als wirksam bei Lag-Merkmalen gezeigt.  
   * Diese transformierten Versionen für das Modelltraining verwenden.

2. **Negative Merkmale separat behandeln.**
   * Für prozentuale Veränderungen und ähnliche Merkmale **Yeo–Johnson** statt log1p einsetzen.  
   * Damit wird eine vollständige Transformationsabdeckung erreicht.

3. **Schiefe nach kombinierter Transformationsstrategie erneut prüfen.**
   * Idealer Bereich: -1 bis +1.  
   * Besonderes Augenmerk auf die 4 Merkmale, die nach log1p noch außerhalb |2| liegen.

4. **Ausreißer-Erkennung nach der Schiefe-Korrektur durchführen.**
   * IQR oder Z-Score für einfachere Merkmale, IsolationForest für multivariate Erkennung.  
   * Validieren, ob extreme Werte reale Ereignisse oder Rauschen darstellen.

5. **Merkmale für ML-Modelle skalieren oder standardisieren.**
   * Besonders wichtig für lineare und distanzbasierte Verfahren.  
   * Für baumbasierte Modelle weniger kritisch, aber dennoch vorteilhaft zur Varianzstabilisierung.

---

### 📊 Zur Datengrundlage

Die erweiterte Analyse bestätigt, dass **Schiefe ein datensatzweites Problem** ist.  
Die Screenshots haben das Problem bei einem Teil der Merkmale gezeigt, die vollständige Tabelle belegt jedoch, dass praktisch alle numerischen Variablen betroffen sind.  
Vor dem ML-Training ist daher eine konsistente Transformations-Pipeline erforderlich.

---

✅ **Zusammenfassung**:  
Der Datensatz weist **systemische Rechtsschiefe** auf, mit extremen Werten in mehreren Merkmalen (bis über Schiefe > 40).  
Die Anwendung von `log1p` reduziert die Schiefe deutlich bei positiven Merkmalen, lässt jedoch Merkmale mit negativen Werten unbehandelt.  
Empfohlen wird eine kombinierte Strategie (`log1p` für positive, `Yeo–Johnson` für negative Merkmale), gefolgt von Ausreißer-Erkennung und Skalierung.  
Dies führt zu einem ausgewogeneren und zuverlässigeren Datensatz für das anschließende ML-Modelltraining.
