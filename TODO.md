# TASKS

## Dorci - 2026-05-01

- [x] projekt scope-jának pontosítása
- [x] cikk fő ötletének leszűkítése
- [x] GitHub repo létrehozása
- [x] alap fájlstruktúra kialakítása
- [x] benchmark implementálása
- [x] egyszerű plotting utility-k megírása
- [x] `main.py` elkészítése a benchmarkhoz
- [x] `main.py` lefuttatása
    - [x] analitikus árfüggvény metszeteinek generálása
    - [x] analitikus árfüggvény 3D felületének generálása
- [x] PINN skeleton

## Dorci - 2026-05-03

- [x] README
- [x] első commit
- [x] GitHub link elküldése a tanárnak

## Dorci - 2026-05-04

- [x] README rendbetétele
- [x] `src/__init__.py` hozzáadása
- [x] PINN bemenetek skálázása
- [x] szintetikus pontmintavételezés megírása
- [x] PINN loss komponenseinek előkészítése
- [x] egyszeri loss-számítás kipróbálása egy még nem tanított modellen
- [x] második commit és push

## TODO

### Black-Scholes benchmark

- [ ] átnézni még egyszer a Black-Scholes képlet implementációját
- [ ] pontosítani, milyen `(t, S)` tartományon dolgozunk
- [ ] létrehozni egy tesztrácsot a kiértékeléshez
- [ ] kiszámolni az analitikus Black-Scholes értékeket ezen a rácson
- [ ] megírni az MSE/MAE kiértékelést

### PINN rész

- [ ] véglegesíteni az egyszerű PyTorch modellt
- [ ] kipróbálni, hogy a `forward(t, S)` tényleg a megfelelő alakú outputot adja
- [ ] megírni a PDE residual számítását autograddal
- [ ] megírni a loss külön részeit:
  - [ ] payoff feltétel
  - [ ] peremfeltételek
  - [ ] PDE residual
- [ ] összerakni a teljes loss-függvényt

### Training és eredmények

- [ ] megírni az első training loopot
- [ ] eltárolni a loss értékeket tanítás közben
- [ ] kirajzolni a loss görbét
- [ ] futtatni egy rövid próba-traininget, hogy látszódjon, működik-e
- [ ] ha stabil, futtatni egy hosszabb tanítást
- [ ] kiszámolni a PINN predikcióját a tesztrácson
- [ ] összehasonlítani az analitikus megoldással
- [ ] hibákat ábrázolni metszeteken és/vagy felületen
- [ ] eredmények rövid értelmezése

### Ha marad idő

- [ ] megnézni, tudunk-e egyszerűen valós opciós adatot szerezni (pl. `yfinance`)
- [ ] esetleg megnézni, tudjuk-e ugyanazt az adathalmazt használni, mint a cikk szerzői?
- [ ] választani egy részvényt és egy lejáratot
- [ ] letölteni egy call opciós láncot
- [ ] bid/ask alapján kiszámolni egy egyszerű mid price-t
- [ ] összevetni a piaci árakat a Black-Scholes/PINN árakkal

### Dokumentáció / prezentáció

- [ ] README frissítése, ha elkészül a PINN rész
- [ ] futtatási lépések ellenőrzése friss környezetben
- [ ] végső ábrák kiválasztása
- [ ] rövid beszámoló megírása
- [ ] prezentáció összerakása, felkészülés