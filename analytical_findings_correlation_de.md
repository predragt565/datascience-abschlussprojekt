## 🛏️ Eurostat Touristische Übernachtungen 2012–2025 (EU10) Trendanalyse

### Analytische Befunde – Korrelations-Heatmap

Beobachtungskategorien:

* Land (`Country`)
* Saison (`Season`)
* NACEr2-Kategorie (`I551 - Hotels` oder `I553 - Campingplätze`)
* Top **Merkmalskorrelationen mit `value`** (Übernachtungen)

---

## 🌍 Globale Kernergebnisse – Erkenntnisse aus der Korrelations-Heatmap (Stichprobe: 6 Länder)

✅ **Stärkste positive Korrelationen**

* **`Lag_1`** ist der mit Abstand zuverlässigste Prädiktor in **allen sechs Ländern**.  
  → Übernachtungen sind **hochgradig autokorreliert**; die letzte Periode ist der beste Indikator für die aktuelle.
* **`MA3`** liegt konsistent auf Platz zwei und übertrifft in **Finnland** gelegentlich sogar `Lag_1`.  
  → Kurzfristige geglättete Trends erfassen saisonale Anstiege und Rückgänge effektiver als längere Durchschnitte.

🔄 **Lag-/Lead-Effekte**

* `Lag_1` > `Lag_3` > `Lag_12` über alle Länder hinweg.
* **Jährliches Gedächtnis (`Lag_12`)** ist am stärksten in **Dänemark und Kroatien**, wo sich jährliche Spitzen und Tiefs zuverlässig wiederholen.
* In **Deutschland und Finnland** sind kurzfristige Kontinuität (`Lag_1`, `MA3`) deutlich wichtiger als die Werte des Vorjahres.

🌱 **Saisonale Muster**

* **Sommer** zeigt immer die **höchsten Korrelationen**, was die Stärke der touristischen Spitzensaison widerspiegelt.
* **Winter**-Korrelationen sind schwächer, aber weiterhin positiv → Nebensaisons bleiben durch Autokorrelation vorhersehbar.
* **Zyklische Encodings** (`Month_cycl_sin`, `Month_cycl_cos`) erfassen Phaseninformationen, tragen aber weniger bei als Lags.

🔽 **Stärkste negative Korrelationen**

* **`Month_cycl_cos`** ist in **5/6 Ländern** das am stärksten negativ korrelierte Merkmal.  
  → Spiegelt **saisonale Täler** wider, wobei der Cosinus die Tiefpunkte zeitlich abbildet.
* **Deutschland** ist die Ausnahme, wo der rohe `Monat`-Wert den stärksten negativen Effekt zeigte.  
  → Deutet auf länderspezifische Eigenheiten hin, wie Saisonalität mit dem Kalender interagiert.

⚠️ **Ausreißer & ungewöhnliche Befunde**

* **Deutschland** – Der rohe `Monat`-Wert ist stärker negativ als die zyklischen Encodings.
* **Finnland** – `MA3` schlägt gelegentlich `Lag_1`, was eine außergewöhnliche Stabilität in geglätteten Trends zeigt.
* **Dänemark & Kroatien** – Der Jahreszyklus (`Lag_12`) ist ungewöhnlich stark und hebt die starke Wiederholung Jahr für Jahr hervor.
* **Spanien & Portugal** – Verhalten sich wie Deutschland und Kroatien, aber mit leicht schwächerer Wintervorhersagbarkeit für Campingplätze.

---

## 🏆 Gesamtrangliste der Merkmale (Durchschnittliche Korrelation mit `value`)

| Rang | Merkmal          | Ø-Korr.   | Interpretation                                                                 |
| ---- | ---------------- | --------- | ------------------------------------------------------------------------------- |
| 🥇 1 | **Lag\_1**       | **0.93**  | Stärkster Prädiktor überall → Übernachtungen sind stark autokorreliert.        |
| 🥈 2 | **MA3**          | **0.87**  | Kurzfristige Glättung erfasst saisonale An- und Abstiege effektiv.             |
| 🥉 3 | **Lag\_12**      | **0.79**  | Erinnerung an den Jahreszyklus; am stärksten in Dänemark & Kroatien.           |
| 4    | MA6              | 0.72      | Mittelfristige Glättung, nützlich aber weniger reaktionsschnell als MA3.       |
| 5    | Lag\_3           | 0.69      | Mittelfristiges Gedächtnis (3 Monate); schwächer als kurzfristige/jährliche.   |
| 6    | MA12             | 0.65      | Langfristige Glättung; stabil, aber weniger vorhersagend.                      |
| 7    | Monat            | –0.42     | Roher Kalendermonat fügt Rauschen hinzu; nur in Deutschland am negativsten.    |
| 8    | Month\_cycl\_cos | –0.39     | Erfasst Tiefpunkte; stärkste negative Korrelation in 5/6 Ländern.              |
| 9    | Month\_cycl\_sin | –0.21     | Erfasst saisonale Phasen; schwächer und weniger konsistent als Cosinus.        |

---

### 🔎 Wichtige Gesamterkenntnisse aus der Rangliste

* **Lagged Features dominieren:**  
  `Lag_1` ist überall #1. `Lag_12` bestätigt die **jährliche Wiederholung**, aber seine Stärke variiert.

* **Gleitende Durchschnitte sind wichtig:**  
  `MA3` ist global auf Platz 2 und schlägt in Finnland manchmal sogar `Lag_1`. Kurze Fenster sind nützlicher als lange (`MA12`).

* **Saisonale Encodings sind schwächer:**  
  Zyklische Merkmale (`Month_cycl_cos`, `Month_cycl_sin`) haben negative Korrelationen und erfassen **Tiefphasen**, aber liefern weniger Vorhersagekraft als Lags/MA.

* **Ausreißer:**  
  Deutschland ist das **einzige Land**, wo der rohe `Monat` das stärkste negative Signal ist → kalendergetriebene Eigenheiten.

---

## 🌍 Was die Analyse wirklich bedeutet (einfach erklärt)
***Wichtig: Diese Ergebnisse bei ML-Modellierung nutzen***

### 1. **Gestern sagt heute voraus**

* Das Merkmal **`Lag_1`** (Übernachtungen des Vormonats) ist überall der stärkste Prädiktor.
* Praxis: Wenn im letzten Monat viele Menschen übernachtet haben, ist die Wahrscheinlichkeit hoch, dass es diesen Monat wieder so ist.
* Übernachtungen folgen **Momentum** — sie ändern sich nicht abrupt, außer bei Schocks (z. B. Krise, Pandemie).

---

### 2. **Kurzfristige Trends zählen**

* Der **3-Monats-Durchschnitt (MA3)** ist fast so stark wie `Lag_1` und in Finnland manchmal sogar besser.
* Praxis: Wenn die letzten drei Monate im Aufwärtstrend waren, wird der nächste Monat fast sicher ebenfalls steigen.
* Glättet „Rauschen“ (schlechtes Wetter-Wochenende, einmaliges Event) und zeigt den **echten saisonalen Trend**.

---

### 3. **Tourismus ist saisonal und wiederholt sich jedes Jahr**

* Der **12-Monats-Lag (`Lag_12`)** bestätigt: Was im **Vorjahr im gleichen Monat** passierte, ist ein starker Prädiktor — besonders in Dänemark und Kroatien.
* Praxis: Wenn Kroatien im letzten August viele Gäste hatte, wird das wahrscheinlich auch dieses Jahr so sein.

---

### 4. **Saisonale Hochs und Tiefs sind sichtbar**

* Merkmale wie **Month\_cycl\_cos** und **Month\_cycl\_sin** sind mathematische Kalenderabbildungen.
* Sie zeigen deutliche **Spitzen (Sommer) und Täler (Winter)** in Übernachtungen.
* Diese Signale sind schwächer als Lags, helfen aber, den **Zeitpunkt der Nachfrageänderung** zu erklären.

---

### 5. **Länderspezifische Eigenheiten**

* Deutschland ist ungewöhnlich: Der **rohe Monatswert** war dort am stärksten negativ (nicht die Encodings).
* Das deutet darauf hin, dass der Tourismus in Deutschland stärker an **konkrete Kalendermonate** (Ferien, Schulferien) gebunden ist als an glatte saisonale Kurven.

---

## 🛎️ Was dies über Übernachtungstrends erklärt

* **Sommer ist König** 🏖️ — in allen Ländern ist die Spitzennachfrage im Sommer und sehr vorhersehbar.
* **Winter ist schwächer** ❄️, folgt aber Mustern (Skitourismus, Weihnachtsmärkte).
* **Frühling und Herbst** 🌱🍂 sind Übergangszeiten — Aufbau in den Sommer, Rückgang danach.
* **Stabilität dominiert**: Tourismusströme schwanken nicht stark, sondern wiederholen sich Jahr für Jahr.

---

## 🔮 Kann dies zur Prognose genutzt werden?

✅ **Ja.**

* Diese Korrelationen zeigen, dass **Prognosemodelle** mit wenigen Merkmalen gebaut werden können:

  * Vormonatswert (`Lag_1`)
  * 3-Monats-Trend (`MA3`)
  * Vorjahreswert (`Lag_12`)
  * Ein saisonales Signal (Monat des Jahres)

* Damit lassen sich **Übernachtungen für die nächste Saison mit solider Sicherheit prognostizieren**.
* Solche Modelle erfassen keine Schocks (COVID, Naturkatastrophen, neue Flugrouten), funktionieren aber **gut für normale Saisonalität**.

---

## 🏳️ Länderspezifische Korrelationstrends

Synthese der Ergebnisse pro Land anhand von:

1. **Stärksten Korrelationen** (positiv/negativ)  
2. **Saisonalen Unterschieden**  
3. **Lag-/Lead-Effekten**  
4. **Bemerkenswerten Ausreißern**

---

## 🇩🇪 Deutschland (DE) – Heatmap-Erkenntnisse

| NACEr2 (Typ)            | Saison   | Stärkste Korrelationen mit `value` | Interpretation                                              |
| ----------------------- | -------- | ---------------------------------- | ----------------------------------------------------------- |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.97, `MA3`: 0.90         | Sehr starke kurzfristige Autokorrelation, glattes Trendsignal |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_1`: 0.98, `MA3`: 0.92         | Spitzensaison sehr gut vorhersehbar aus jüngsten Werten     |
| Hotels, Gasthöfe (I551) | Herbst   | `Lag_1`: 0.96, `MA3`: 0.88         | Herbstlicher Rückgang folgt weiterhin Historie + Glättung   |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_1`: 0.95, `MA3`: 0.87         | Nebensaison stabil, Kontinuität bleibt wichtig              |
| Campingplätze (I553)    | Frühling | `Lag_1`: 0.96, `MA3`: 0.87         | Frühlingserholung gut durch kurzfristige + Glättung erfasst |
| Campingplätze (I553)    | Sommer   | `Lag_1`: 0.98, `MA3`: 0.92         | Extrem hohe Kontinuität in der Spitzensaison                |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.95, `MA3`: 0.84         | Saisonende folgt weiter Momentum                            |
| Campingplätze (I553)    | Winter   | `Lag_1`: 0.94, `MA3`: 0.83         | Winterwerte vorhersagbar aus jüngsten Trends                |

---

### 🔑 Wichtige Erkenntnisse (Deutschland)

✅ **Stärkste Merkmalskorrelationen**

* Über alle Slices dominieren `Lag_1` und `MA3`.
* **🔼 Stärkste positive Korrelation:** `Lag_1` → **0.977** → Monatliche Übernachtungen sind stark autokorreliert.
* **🔽 Stärkste negative Korrelation:** `Monat` → **–0.527** → Roher Monatsindex negativ; zyklische Encodings erfassen die Saisonalität besser.

🌱 **Saisonale Muster**

* **Sommer** zeigt die **höchsten Korrelationen** → starkes Saisonal-Momentum.
* **Winter** schwächer, aber weiterhin vorhersehbar.
* `Month_cycl_sin`: –0.256 → erfasst die saisonale Phase, aber moderat.
* `Month_cycl_cos`: –0.106 → schwache saisonale Cosinus-Komponente.

🔄 **Lag-/Lead-Effekte**

* `Lag_1`: **0.939** → sehr starke kurzfristige Persistenz.
* `Lag_3`: 0.733 → mittelfristiges Gedächtnis, weniger Einfluss.
* `Lag_12`: 0.834 → Vorjahreswerte bleiben informativ.
* `MA3` > `MA12` → kurze Durchschnitte erfassen Dynamiken besser.

⚠️ **Ausreißer oder ungewöhnliche Befunde**

* Keine – Deutschlands Muster sind stabil und saisonal.

---

## 🇩🇰 Dänemark (DK) – Korrelations-Heatmap-Erkenntnisse

| NACEr2 (Typ)            | Saison   | Stärkste Korrelationen mit `value` | Interpretation                                               |
| ----------------------- | -------- | ---------------------------------- | ------------------------------------------------------------ |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.83, `MA3`: 0.66         | Frühlings-Momentum beobachtet, jüngste Werte prägend         |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_12`: 0.93, `Lag_1`: 0.91      | Jahreszyklus und kurzfristige Kontinuität beide stark        |
| Hotels, Gasthöfe (I551) | Herbst   | `Lag_1`: 0.76, `MA3`: 0.60         | Abflachen nach dem Sommer, dennoch etwas Kontinuität         |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_12`: 0.92, `Lag_1`: 0.85      | Winter durch Jahresgedächtnis und jüngste Werte erklärt      |
| Campingplätze (I553)    | Frühling | `Lag_1`: 0.70, `MA3`: 0.60         | Kurzfristeffekte dominieren, milder saisonaler Aufbau        |
| Campingplätze (I553)    | Sommer   | `Lag_12`: 0.91, `Lag_1`: 0.83      | Sommerpeak vom Jahreszyklus dominiert, jüngste Nachfrage folgt |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.76, `MA3`: 0.58         | Rückgang vom Sommerpeak, Kontinuität schwächer               |
| Campingplätze (I553)    | Winter   | `Lag_12`: 0.90, `Lag_1`: 0.82      | Nebensaison durch jährliche Wiederholung + Kurzfristgedächtnis getrieben |

---

### 🔑 Wichtige Erkenntnisse (Dänemark)

✅ **Stärkste Merkmalskorrelationen**

* `Lag_12` und `Lag_1` dominieren und spiegeln **Jahreszyklen und kurzfristige Kontinuität** wider.
* **🔼 Stärkste positive Korrelation:** `Lag_12` → **0.933** → jährliche Saisonalität stärkster Treiber.
* **🔽 Stärkste negative Korrelation:** `Month_cycl_cos` → **–0.739** → saisonaler Cosinus erfasst starke Wintertiefs.

🌱 **Saisonale Muster**

* Sommer und Winter stark mit dem Jahreszyklus verknüpft.
* Frühling/Herbst schwächer, Übergang.
* `Month_cycl_sin`: –0.352, `Month_cycl_cos`: –0.163 → klare zyklische Signatur.

🔄 **Lag-/Lead-Effekte**

* `Lag_1`: 0.699 → moderates Kurzfristgedächtnis.
* `Lag_3`: 0.501 → schwächeres Mittelfristgedächtnis.
* `Lag_12`: 0.726 → starker Jahres­effekt.

⚠️ **Ausreißer oder ungewöhnliche Befunde**

* Geringere Vorhersagbarkeit bei Campingplätzen in Frühling/Herbst.

---

## 🇫🇮 Finnland (FI) – Korrelations-Heatmap-Erkenntnisse

| NACEr2 (Typ)            | Saison   | Stärkste Korrelationen mit `value` | Interpretation                                                 |
| ----------------------- | -------- | ---------------------------------- | -------------------------------------------------------------- |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.95, `MA3`: 0.86         | Frühlingswachstum mit Momentum, Trendglättung wirksam          |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_1`: 0.98, `MA3`: 0.96         | Sehr stabile Spitzennachfrage, Historie sagt gut voraus        |
| Hotels, Gasthöfe (I551) | Herbst   | `MA3`: 0.98, `Lag_1`: 0.93         | Geglätteter Trend etwas stärker als reine Kontinuität          |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_1`: 0.94, `MA3`: 0.86         | Winter stabil, momentumgetrieben mit geglätteter Verstärkung   |
| Campingplätze (I553)    | Frühling | `MA3`: 0.99, `Lag_1`: 0.95         | Außergewöhnlich glatter Trend dominiert, starkes Kurzfristgedächtnis |
| Campingplätze (I553)    | Sommer   | `Lag_1`: 0.97, `MA3`: 0.93         | Spitzensaison: starke Kontinuität, geglätteter Trend           |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.94, `MA3`: 0.85         | Allmählicher saisonaler Rückgang, jüngste Werte prädiktiv      |
| Campingplätze (I553)    | Winter   | `Lag_1`: 0.93, `MA3`: 0.84         | Winter-Nebensaison, Momentum bleibt Haupttreiber               |

---

### 🔑 Wichtige Erkenntnisse (Finnland)

✅ **Stärkste Merkmalskorrelationen**

* `MA3` oft stärker als reines `Lag_1`.
* **🔼 Stärkste positive Korrelation:** `MA3` → **0.989** → geglättete Trends dominieren.
* **🔽 Stärkste negative Korrelation:** `Month_cycl_cos` → **–0.528** → Cosinus-Encoding erfasst Off-Peaks.

🌱 **Saisonale Muster**

* Sehr stabil über die Saisons hinweg.
* `Month_cycl_sin`: –0.096, `Month_cycl_cos`: –0.106 → schwache zyklische Encodings im Vergleich zu Lags.

🔄 **Lag-/Lead-Effekte**

* `Lag_1`: 0.839, `Lag_3`: 0.753, `Lag_12`: 0.807 → sowohl kurzfristiges als auch jährliches Gedächtnis stark.

⚠️ **Ausreißer oder ungewöhnliche Befunde**

* Keine – Finnland ist das stabilste der Länder.

---

## 🇭🇷 Kroatien (HR) – Korrelations-Heatmap-Erkenntnisse

| NACEr2 (Typ)            | Saison   | Stärkste Korrelationen mit `value` | Interpretation                                            |
| ----------------------- | -------- | ---------------------------------- | --------------------------------------------------------- |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.93, `MA3`: 0.81         | Frühjahrsaufbau mit Kurzfristgedächtnis + Glättung        |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_1`: 0.96, `MA3`: 0.91         | Sehr starke Vorhersagbarkeit der Spitzensaison            |
| Hotels, Gasthöfe (I551) | Herbst   | `Lag_1`: 0.91, `MA3`: 0.73         | Herbstlicher Rückgang weiterhin momentumgetrieben         |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_1`: 0.89, `MA3`: 0.70         | Winter stabil, aber schwächer als Spitzenperioden         |
| Campingplätze (I553)    | Frühling | `Lag_1`: 0.93, `MA3`: 0.81         | Frühjahrs­erholung, Momentum + Trendglättung              |
| Campingplätze (I553)    | Sommer   | `Lag_1`: 0.97, `MA3`: 0.90         | Extrem starke Sommer‑Autokorrelation, Spitzentourismus    |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.92, `MA3`: 0.77         | Saisonausklang, Momentum weiterhin vorhanden              |
| Campingplätze (I553)    | Winter   | `Lag_1`: 0.88, `MA3`: 0.68         | Winter-Nebensaison schwächer, Vergangenheit bleibt prädiktiv |

---

### 🔑 Wichtige Erkenntnisse (Kroatien)

✅ **Stärkste Merkmalskorrelationen**

* `Lag_1` durchgängig Top-Prädiktor.
* **🔼 Stärkste positive Korrelation:** `Lag_1` → **0.962** → Kontinuität dominiert.
* **🔽 Stärkste negative Korrelation:** `Month_cycl_cos` → **–0.563** → Cosinus-Encoding spiegelt saisonale Tiefpunkte.

🌱 **Saisonale Muster**

* Sommer-Autokorrelation extrem hoch.
* `Month_cycl_sin`: –0.336, `Month_cycl_cos`: –0.178 → moderater zyklischer Effekt.

🔄 **Lag-/Lead-Effekte**

* `Lag_1`: 0.930, `Lag_3`: 0.499, `Lag_12`: 0.826 → kurzfristig deutlich stärker als mittelfristig.

⚠️ **Ausreißer oder ungewöhnliche Befunde**

* Winter-Campingplätze schwächer als Hotels.

---

## 🇵🇹 Portugal (PT) – Korrelations-Heatmap-Erkenntnisse

| NACEr2 (Typ)            | Saison   | Stärkste Korrelationen mit `value` | Interpretation                                        |
| ----------------------- | -------- | ---------------------------------- | ----------------------------------------------------- |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.94, `MA3`: 0.87         | Klarer Frühjahrsaufbau, getrieben von jüngsten Anstiegen |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_1`: 0.98, `MA3`: 0.92         | Äußerst stabile Sommernachfrage, gut vorhersagbar     |
| Hotels, Gasthöfe (I551) | Herbst   | `Lag_1`: 0.93, `MA3`: 0.84         | Herbstkontinuität, geglätteter Trend wichtig          |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_1`: 0.91, `MA3`: 0.80         | Winter-Nebensaison stabil, jüngste Werte zählen       |
| Campingplätze (I553)    | Frühling | `Lag_1`: 0.93, `MA3`: 0.80         | Sanfte Frühjahrs­erholung                             |
| Campingplätze (I553)    | Sommer   | `Lag_1`: 0.98, `MA3`: 0.91         | Sehr hohe Sommerkontinuität, Autokorrelation dominiert |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.93, `MA3`: 0.76         | Saisonausklang, weiterhin momentumgetrieben           |
| Campingplätze (I553)    | Winter   | `Lag_1`: 0.90, `MA3`: 0.78         | Winter stabil, jedoch etwas schwächer                 |

---

### 🔑 Wichtige Erkenntnisse (Portugal)

✅ **Stärkste Merkmalskorrelationen**

* `Lag_1` dominiert, `MA3` durchgehend unterstützend.
* **🔼 Stärkste positive Korrelation:** `Lag_1` → **0.981** → kurzfristige Persistenz am stärksten.
* **🔽 Stärkste negative Korrelation:** `Month_cycl_cos` → **–0.730** → Cosinus erfasst Tiefpunkte stark.

🌱 **Saisonale Muster**

* Sommer mit höchsten Korrelationen, sehr stabil.
* `Month_cycl_sin`: –0.302, `Month_cycl_cos`: –0.118 → schwacher bis moderater zyklischer Beitrag.

🔄 **Lag-/Lead-Effekte**

* `Lag_1`: 0.910, `Lag_3`: 0.440, `Lag_12`: 0.676 → kurzfristig deutlich stärker als mittel/jährlich.

⚠️ **Ausreißer oder ungewöhnliche Befunde**

* Keine signifikanten – Portugal folgt den Mustern von Deutschland/Kroatien.

---

## 🇪🇸 Spanien (ES) – Korrelations-Heatmap-Erkenntnisse

| NACEr2 (Typ)            | Saison   | Stärkste Korrelationen mit `value` | Interpretation                                            |
| ----------------------- | -------- | ---------------------------------- | --------------------------------------------------------- |
| Hotels, Gasthöfe (I551) | Frühling | `Lag_1`: 0.94, `MA3`: 0.84         | Sehr starker Kurzfristeffekt, sanfter saisonaler Übergang |
| Hotels, Gasthöfe (I551) | Sommer   | `Lag_1`: 0.99, `MA3`: 0.89         | Spitzensaison-Konsistenz, sehr gut vorhersagbar           |
| Hotels, Gasthöfe (I551) | Herbst   | `Lag_1`: 0.93, `MA3`: 0.70         | Starke Autokorrelation und Einfluss jüngster Trends       |
| Hotels, Gasthöfe (I551) | Winter   | `Lag_1`: 0.92, `MA3`: 0.75         | Winter stabil, Kurzfrist-Momentum                         |
| Campingplätze (I553)    | Frühling | `Lag_1`: 0.91, `MA3`: 0.80         | Frühjahrs­erholung getrieben von jüngster Aktivität       |
| Campingplätze (I553)    | Sommer   | `Lag_1`: 0.90, `MA3`: 0.73         | Hohe Kontinuität, sommerliche Spitzennachfrage            |
| Campingplätze (I553)    | Herbst   | `Lag_1`: 0.89, `MA3`: 0.67         | Herbstlicher Rückgang, jüngste Werte weiter prägend       |
| Campingplätze (I553)    | Winter   | `Lag_1`: 0.87, `MA3`: 0.65         | Winter schwächer, Kurzfristmuster bleiben informativ      |

---

### 🔑 Wichtige Erkenntnisse (Spanien)

✅ **Stärkste Merkmalskorrelationen**

* `Lag_1` ist in allen Slices am stärksten.
* **🔼 Stärkste positive Korrelation:** `Lag_1` → **0.986** → Kontinuität dominiert.
* **🔽 Stärkste negative Korrelation:** `Month_cycl_cos` → **–0.829** → Cosinus-Encoding bildet saisonale Täler scharf ab.

🌱 **Saisonale Muster**

* Sommer-Spitzenwerte nahezu perfekt autokorreliert.
* `Month_cycl_sin`: –0.363, `Month_cycl_cos`: –0.140 → moderates zyklisches Signal.

🔄 **Lag-/Lead-Effekte**

* `Lag_1`: 0.881, `Lag_3`: 0.398, `Lag_12`: 0.639 → kurzfristig deutlich stärker.

⚠️ **Ausreißer oder ungewöhnliche Befunde**

* Campingplätze im Winter geringfügig schwächer als Hotels.

---
