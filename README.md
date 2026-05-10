# Black-Scholes Option Pricing with Physics-Informed Neural Networks

## Rövid projektleírás

Alapötlet: https://arxiv.org/abs/2312.06711

A projektben azt vizsgáljuk, hogyan lehet a Black-Scholes opcióárazási problémát Physics-Informed Neural Network (PINN) segítségével megközelíteni.

Nem a teljes cikk reprodukciója a cél, hanem annak egy kisebb, jól körülhatárolható részét szeretnénk megvalósítani. Első körben egy kontrollált európai call opciós példán dolgozunk. Ebben az esetben ismert az analitikus Black-Scholes megoldás, ezért pontosan látjuk, hogy a PINN által tanult árfüggvény mennyire közelíti a várt eredményt. Ez jó kiindulópont a PINN implementáció felépítéséhez és teszteléséhez.

A projekt alapját egy szintetikus példa adja, nem piaci adathalmaz. A PINN tanításához a Black-Scholes PDE értelmezési tartományából választunk pontokat, az eredményeket pedig az analitikus Black-Scholes képlettel hasonlítjuk össze. Ha marad idő, megnézünk valódi piaci opciós adatokat is.

## Matematikai háttér

A Black-Scholes PDE (parciális differenciálegyenlet):

```math
\frac{\partial V}{\partial t}
+
\frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+
rS\frac{\partial V}{\partial S}
-
rV
=
0.
```

ahol:

- `S`: részvényárfolyam,
- `t`: idő,
- `V(t,S)`: az opció ára,
- `K`: kötési árfolyam / strike price,
- `T`: lejárati idő,
- `r`: kockázatmentes kamatláb,
- `sigma`: volatilitás.

Európai call opció lejáratkori feltétele:

```math
V(T,S) = \max(S-K,0).
```

## Projektterv

### 1. Analitikus benchmark

Implementálom az európai call opció Black-Scholes képletét, majd kirajzolom az árfüggvényt különböző időpontokban.

### 2. PINN-modell

PyTorch segítségével neurális hálót építek. Az első benchmarkban a modell bemenete `(t, S)`, kimenete pedig `V(t,S)`. A valós opciós adatok előkészítéséhez elkezdtem egy parametrikus PINN-változatot is, ahol a strike price is bemenet: `(t, S, K)`.

### 3. Loss-függvény

A loss három fő részből áll majd:

1. lejáratkori feltétel hibája,
2. peremfeltételek hibája,
3. Black-Scholes PDE residual hibája.

### 4. Kiértékelés

A PINN eredményét az analitikus Black-Scholes árral hasonlítjuk majd össze. A tervezett kiértékelési elemek:

- MSE és MAE a PINN-predikció és az analitikus ár között,
- a PINN tanítási veszteségének görbéje,
- analitikus és prediktált árfelület,
- hibafelület,
- metszetek fix időpontokban.

## Jelenlegi státusz

- [x] projekt scope és fókusz meghatározva
- [x] analitikus Black-Scholes benchmark implementálva
- [x] benchmark-ábrák generálása megvan
- [x] PINN alapkomponensek előkészítve
- [x] első PINN training loop elkészült
- [x] rövid próba-training és loss curve generálva
- [x] PINN-predikció kiértékelése az analitikus benchmarkhoz képest
- [ ] hibák vizualizálása és eredmények értelmezése
- [ ] ha van idő: valós piaci adatos kiegészítés
- [ ] végső dokumentáció és prezentáció elkészítése

## Futtatás

Virtuális környezet létrehozása:

```bash
python -m venv .venv
```

Aktiválás Windows PowerShell alatt:

```powershell
.venv\Scripts\Activate.ps1
```

Aktiválás macOS/Linux alatt:

```bash
source .venv/bin/activate
```

Csomagok telepítése:

```bash
pip install -r requirements.txt
```

Analitikus Black-Scholes benchmark futtatása:

```bash
python main.py
```

PINN komponensek gyors ellenőrzése:

```bash
python check_pinn_components.py
```

A generált ábrák a `figures/` mappába kerülnek.

PINN első rövid tanításának futtatása:

```bash
python train_pinn.py
```

Ez elmenti a tanítási veszteséggörbét:

```text
figures/pinn_training_loss.png
```

PINN kiértékelése az analitikus benchmarkhoz képest:

    python evaluate_pinn.py

Ez elmenti az összehasonlító és hibaábrákat:

    figures/pinn_vs_analytic_slices.png
    figures/pinn_error_surface.png

Valós opciós adatok letöltésének és első összehasonlításának kipróbálása:

    python real_data_demo.py

Ez egy Yahoo Finance / yfinance alapú demó, amely egy kiválasztott részvény call opciós láncát tölti le, bid/ask alapján mid price értéket számol, majd összehasonlítja ezeket az analitikus Black-Scholes árakkal.

Parametrikus PINN komponensek gyors ellenőrzése:

    python check_parametric_pinn.py

## Forrás

A projekt alapötlete a következő cikkből származik:

Dhiman, A. and Hu, Y. (2023) Physics informed Neural Network for option pricing, arXiv.org. Available at: https://arxiv.org/abs/2312.06711 (Accessed: 03 May 2026). 

## Használt / tervezett programcsomagok és eszközök
## TODO: véglegesítés (majd a projekt végén)

- Python
- NumPy
- SciPy
- Matplotlib
- PyTorch
- tqdm
- Jupyter Notebook
- Git/GitHub