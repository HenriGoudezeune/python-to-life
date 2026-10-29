# Game of Life + een compiler van Python naar Life

Dit project bevat drie dingen:

1. Een snelle **Game of Life-simulator** in Python (bit-packed numpy, alleen actieve tegels).
2. Een **viewer** in HTML/JS, plus laden en opslaan in **RLE**.
3. Een **compiler** die een kleine Python-functie omzet naar een Life-startpatroon.
   - Het patroon rekent de functie uit met botsende gliders.
   - Het resultaat verschijnt als gliders op vaste uitgangsbanen.
   - Elke poort en elk programma is door simulatie gecontroleerd tegen Python zelf.

## Installatie

```bash
pip install numpy scipy pytest
```

## Snel aan de slag

```bash
# 4-bit opteller: compileren, simuleren, vergelijken met Python, viewer openen
python -m gol.compiler examples/add4.py --inputs a=5,b=9 --run --drc --html
```

Uitvoer:

```
compiled add4 in 0.3s: {'NOR': 43, 'FAN': 9, 'gliders': 91, 'statics': 235, ...}
DRC: clean
wrote out/add4_a5_b9.rle (outputs readable at generation 34377)
Game of Life: 14   Python: 14   [OK] (34377 generations, 12s)
```

In de viewer:
- Labels tonen de invoergliders (`in a[0]` …), de poorten (`NOR#3`, `FAN#5`) en de uitgangen (`out out[0]` …).
- Typ bij "go to generation" het getal uit `outputs readable at generation` in, en lees af welke uitgangsgliders er vliegen.

Andere voorbeelden staan in `examples/`: `logic.py` (majority), `max2.py`, `popcount.py`.

### Als bibliotheek

```python
from gol.compiler import compile_function

prog = compile_function('''
def f(a: u4, b: u4) -> u5:
    return a + b
''')
cells = prog.cells(a=5, b=9)        # (N,2)-array met (x, y) van levende cellen
open("f.rle", "w").write(prog.rle(a=5, b=9))
r = prog.run(a=5, b=9)              # simuleren
print(prog.result(r.outputs), r.clean)   # 14 True
```

### Simulator en viewer

```python
from gol import life, rle, patterns
u = life.Universe(patterns.GOSPER_GUN)
u.step(1000)
rle.save("gun.rle", u.cells)
```

```bash
python -m gol.view gun.rle          # schrijft gun.html (RLE ingebed) en opent hem
```

Je kunt `viewer/index.html` ook rechtstreeks openen. Daarin kun je een RLE laden (bestand, drag-and-drop of plakken), afspelen, de snelheid kiezen, zoomen en pannen, en naar een generatie springen.

## Ondersteunde Python-subset

```python
def naam(a: u4, b: u4, c: bool) -> u5:   # elke parameter: uN (N bits, unsigned) of bool
    ...
```

- **Types**: alleen unsigned bitvectoren. Constanten zijn niet-negatieve ints of bools.
- **Operatoren**:
  - `+` groeit met één bit, `*` telt de breedtes op.
  - `-`, unaire `-` en `~` wrappen binnen de breedte van de operanden. `~5` op een u4 geeft dus 10, niet −6 zoals in Python.
  - `& | ^`
  - `<< >>`, ook met variabele schuifafstand.
  - `//` en `%` alleen door constante machten van 2.
  - Vergelijkingen (ook gekoppeld, `a < b < c`).
  - `and/or/not`. Die geven altijd een bool terug.
  - `x if c else y`.
- **Functies**: `min`, `max`, `abs`, `int`, `bool`.
- **Statements**:
  - Toewijzing (ook `a, b = ...`), `+=` en andere samengestelde toewijzingen.
  - `if/elif/else`.
  - `for i in range(...)` met constante grenzen. De lus wordt uitgerold.
  - `return` overal, ook vroeg in een `if` of lus.
- **Uitvoer**: de return-annotatie `-> uN` bepaalt de breedte. Een tuple-return geeft meerdere uitgangen (`out0`, `out1`, …).

Niet ondersteund: `while`, lijsten, strings, floats, aanroepen van eigen functies, negatieve getallen, `/`, `break`/`continue`.

## Hoe het werkt (kort; details in CLAUDE.md)

1. **Frontend**: `ast` → and-inverter-graph (AIG) met structurele hashing. Optellers, vergelijkers, muxen en schuifoperaties worden op bitniveau opgebouwd.
2. **Techmap**:
   - Elke AND wordt een NOR van de complementen.
   - Signalen die meer dan eens gebruikt worden, krijgen een goedkoopste boom van FAN-poorten (1 in, 2× NOT uit) en NOT's.
3. **Layout**:
   - Elk signaal is één glider (aanwezig = 1). Signalen lopen als SE-gliders ("rijen").
   - Elke poort is een constante SW-glider ("kolom") die alle rijen kruist. Hij botst alleen met zijn eigen invoer, omdat de timing (θ) van alle andere rijen minstens 19 generaties afwijkt.
   - Na de poort brengt een vertragingslus van drie boten (omlaag, rechts, omhoog, rechts) de uitgang naar een nieuwe rij met precies de timing die de volgende poort nodig heeft.
4. **Reacties** komen uit een brute-force botsingszoektocht (`python -m gol.search`):
   - glider+glider-annihilatie voor NOT/NOR;
   - kickback voor fanout;
   - boot als eenmalige reflector;
   - eater1.

   Geen enkele timing komt uit het geheugen: alles is gesimuleerd.
5. **Controle**:
   - `drc` zoekt geometrisch naar onbedoelde ontmoetingen.
   - `verify` simuleert en eist dat alleen de verwachte uitgangsgliders overblijven.

## Bewijs dat het werkt

```bash
python -m pytest -q -m "not slow"    # snel (simulator, reacties, poorten, frontend)
python -m pytest -q                  # alles, inclusief de trage opteller-tests
```

- `tests/test_life.py`: de simulator tegen een naïeve referentie, gliderbeweging, Gosper-gun met periode 30, RLE.
- `tests/test_reactions.py`: elke gebruikte reactie opnieuw gesimuleerd.
- `tests/test_gates.py`: NOT, NOR2, NOR3, FAN en een poortketen, exhaustief in Life gesimuleerd.
- `tests/test_frontend.py`: Python-semantiek van de frontend en de techmap tegen echte Python.
- `tests/test_programs.py`: hele programma's in Life tegen Python.
  - Exhaustief: majority, xor3, 2-bit opteller, popcount.
  - Steekproef: max2 en de 4-bit opteller.

## Beperkingen

- **Eenmalig**: het patroon rekent één keer. Voor andere invoer maak je een nieuw beginpatroon (`prog.cells(...)`). Er zijn geen guns en geen klok.
- **Invoercodering**: een invoerbit bepaalt of één of meer gliders aanwezig zijn. Soms gaat het om gecomplementeerde gliders, die aanwezig zijn als de bit 0 is.
- **Grootte**: de patronen worden snel groot.
  - De 4-bit opteller is ongeveer 11 000 × 8 500 cellen en geeft zijn antwoord na ongeveer 34 000 generaties. Dat is zo'n 12 s simuleren.
  - Elke poort krijgt een eigen tijdslot, dus de duur groeit lineair met het aantal poorten.
- **Semantiek**: ints zijn unsigned en wrappen (zie `~` en `-` hierboven). Programma's die negatieve tussenresultaten vergelijken, geven dus een ander resultaat dan Python.
- **Uitvoer**: de uitkomst staat in gliders die wegvliegen. Er is geen blijvende display.
