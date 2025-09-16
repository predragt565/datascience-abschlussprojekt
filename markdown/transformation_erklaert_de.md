
## Zwei unterschiedliche Ziele

1. **Schiefe-Korrektur (Feature-Transformation)**  
   - Ziel: numerische Verteilungen symmetrischer machen, Varianz stabilisieren.  
   - Typische Methoden: `log1p`, Box-Cox, Yeo–Johnson.  
   - Wirkung: verbessert die Leistung von Modellen, die empfindlich auf Verteilungsformen reagieren (Lineare Regression, Lasso, Ridge, ElasticNet usw.).

2. **Ausreißer-Erkennung (Zeilenfilterung)**  
   - Ziel: gesamte Zeilen entfernen oder gewichten, die im Vergleich zu den übrigen stark abweichen.  
   - Methoden: IQR, Z-Score, IsolationForest.  
   - Wirkung: verhindert, dass wenige extreme Beobachtungen das Training dominieren.

---

### Sollten beide Schritte genutzt werden?  
Ja — jedoch sind sie nicht austauschbar, und die Reihenfolge ist entscheidend:

1. **Zuerst Schiefe-Transformation anwenden (log1p oder Yeo–Johnson).**  
   - Reduziert lange Verteilungsschwänze und macht die Ausreißer-Erkennung zuverlässiger.  
   - Beispiel: Wird Z-Score auf stark schiefe Rohdaten angewendet, sind Mittelwert und Standardabweichung durch die langen Schwänze verzerrt → zu viele Punkte werden fälschlich als Ausreißer markiert.

2. **Anschließend Ausreißer-Erkennung durchführen (IQR, Z-Score oder IsolationForest).**  
   - Die Verteilungen sind nun näher an der Normalverteilung → Schwellenwerte (wie ±3 Standardabweichungen) sind sinnvoller.  
   - Ob erkannte Zeilen entfernt oder beibehalten werden, hängt vom Fachwissen ab.

---

### Welche Transformation für den ML-Pipeline nutzen?
- **Log1p**: Einfach, geeignet für ausschließlich positive Merkmale (z. B. Zählwerte, Geldbeträge, Zeitdauern).  
- **Yeo–Johnson**: Flexibler (unterstützt auch 0 und negative Werte). Falls der Datensatz negative/Nullwerte enthält, ist Yeo–Johnson sicherer.  

Da die betrachteten Merkmale häufig Zählwerte/Aggregate (≥ 0) darstellen, ist **log1p ausreichend und gut interpretierbar**.  
Eine Kombination von log1p und Yeo–Johnson ist nicht notwendig. Eine Methode genügt.

---

### Empfohlener Workflow

1. **log1p** anwenden (oder Yeo–Johnson, falls nur eine Transformation vorgesehen ist).  
2. **Ausreißer-Erkennung** auf den transformierten Daten durchführen (eine Methode auswählen):  
   - Z-Score → einfach, funktioniert gut nach Schiefe-Korrektur.  
   - IQR → robust, weniger abhängig von Verteilungsannahmen.  
   - IsolationForest → geeignet für komplexe/hohe Dimensionen, jedoch weniger interpretierbar.  
3. ML-Modell mit dem **bereinigten + transformierten Datensatz** trainieren.  

---

### Schlussbemerkungen
- Schiefe-Transformation und Ausreißer-Erkennung lösen unterschiedliche Probleme.  
- Beide sollten genutzt werden, in der richtigen Reihenfolge:  
  - Transformation (log1p oder Yeo–Johnson) → zuerst.  
  - Ausreißer-Erkennung (Z-Score / IQR / IF) → danach.  
- Für Datensätze mit stark rechtsschiefen positiven Werten ist **log1p + Z-Score** häufig eine wirksame und effiziente Wahl.  
- Das Entfernen von Ausreißern sollte durch den Kontext geleitet sein: extreme Spitzen können valide Ereignisse darstellen und wertvolle Vorhersageinformationen enthalten.
