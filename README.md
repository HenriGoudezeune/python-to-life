# python-to-life

Een **compiler van Python naar Conway's Game of Life**. Je schrijft een kleine Python-functie. De compiler maakt er een Life-startpatroon van dat die functie uitrekent met botsende gliders. Het resultaat verschijnt als gliders op vaste uitgangsbanen.

Elke poort en elk programma is door simulatie gecontroleerd tegen Python zelf. Ook de botsingsreacties waarop alles steunt, zijn door simulatie gevonden en niet uit een handboek overgenomen.

```python
def add4(a: u4, b: u4) -> u5:
    return a + b
```

```bash
python -m gol.compiler examples/add4.py --inputs a=5,b=9 --run --drc --html
```

```
compiled add4 in 0.3s: {'NOR': 43, 'FAN': 9, 'gliders': 91, 'statics': 235, ...}
DRC: clean
wrote out/add4_a5_b9.rle (outputs readable at generation 34377)
Game of Life: 14   Python: 14   [OK] (34377 generations, 12s)
```

`--html` opent het patroon in de meegeleverde viewer. Labels tonen daar de invoergliders (`in a[0]` …), de poorten (`NOR#3`, `FAN#5`) en de uitgangen (`out out[0]` …). Spring naar de generatie uit `outputs readable at generation` en lees af welke uitgangsgliders er vliegen.

## Installatie

```bash
pip install numpy scipy pytest
```

## Gebruik

### Editor (de makkelijkste manier)

```bash
python -m gol.editor
```

Dit opent http://127.0.0.1:8765 in je browser, met links een code-editor en rechts de viewer.
1. Schrijf een functie, of kies een voorbeeld.
2. Klik **Compileer** (`Ctrl+Enter`). Er verschijnen velden voor de invoer, en het patroon wordt in de viewer geladen.
3. Klik **▶ Speel af** (`Ctrl+Shift+Enter`). De viewer speelt het patroon live af, in 10, 30 of 90 seconden, en stopt vanzelf op de generatie waarop de uitgangen af te lezen zijn. Op het einde leest de editor de uitgangsgliders uit de levende cellen, geeft ze een label en zet de uitkomst in Life naast die van Python. Zoom tijdens het afspelen in op een poort (`NOR#…`, `FAN#…`) om de botsingen te zien.
4. Fouten verschijnen met hun regelnummer, dat ook in de kantlijn gemarkeerd wordt.

De server luistert alleen op je eigen computer.

### Command line

```bash
python -m gol.compiler bestand.py [--func naam] [--inputs a=5,b=9] [-o uit.rle] [--run] [--drc] [--html]
```

- `--inputs`: de invoerwaarden. Zonder deze optie zijn alle invoerbits 0.
- `--run`: simuleren en de uitkomst vergelijken met wat Python zelf berekent.
- `--drc`: geometrische controle op onbedoelde ontmoetingen tussen gliders.
- `--html`: een HTML-pagina met de viewer en het patroon schrijven en openen.

Voorbeelden staan in `examples/`: `add4.py`, `logic.py` (majority), `max2.py`, `popcount.py` en `seg7.py`.

`seg7.py` is het spektakelstuk: een 7-segmentdecoder van ongeveer 230 poorten, 360 gliders en 1300 stillevens, 51 000 × 40 000 cellen groot, die rekent tot generatie 162 097. Speel hem af in de editor met de duur op 90 s.

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

## Ondersteunde Python-subset

```python
def naam(a: u4, b: u4, c: bool) -> u5:   # elke parameter: uN (N bits, unsigned) of bool
    ...
```

- **Types**: alleen unsigned bitvectoren (tot 64 bits). Constanten zijn niet-negatieve ints of bools.
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
  - `for i in range(...)` met constante grenzen. De lus wordt uitgerold. Negatieve constanten mogen alleen hier, als argument van `range`.
  - `return` overal, ook vroeg in een `if` of lus.
- **Uitvoer**: de return-annotatie `-> uN` bepaalt de breedte. Een tuple-return geeft meerdere uitgangen (`out0`, `out1`, …). De annotatie moet evenveel elementen hebben als de tuple.

Niet ondersteund: `while`, lijsten, strings, floats, aanroepen van eigen functies, negatieve constanten, `/`, `break`/`continue`. De compiler geeft dan een `CompileError` met regelnummer.

## Hoe het werkt

Details staan in [CLAUDE.md](CLAUDE.md).

1. **Frontend** (`gol/compiler/frontend.py`): `ast` → and-inverter-graph (AIG) met structurele hashing. Optellers, vergelijkers, muxen en schuifoperaties worden op bitniveau opgebouwd.
2. **Techmap** (`techmap.py`):
   - Elke AND wordt een NOR van de complementen.
   - Signalen die meer dan eens gebruikt worden, krijgen een goedkoopste boom van FAN-poorten (1 in, 2× NOT uit) en NOT's.
3. **Layout** (`layout.py`):
   - Elk signaal is één glider (aanwezig = 1). Signalen lopen als SE-gliders ("rijen").
   - Elke poort is een constante SW-glider ("kolom") die alle rijen kruist. Hij botst alleen met zijn eigen invoer, omdat de timing (θ) van alle andere rijen minstens 19 generaties afwijkt.
   - Na de poort brengt een vertragingslus van drie reflectoren de uitgang naar een nieuwe rij, met precies de timing die de volgende poort nodig heeft.
4. **Reacties** (`gol/search.py`, `gol/reactions.py`) komen uit een brute-force botsingszoektocht:
   - glider+glider-annihilatie voor NOT/NOR;
   - kickback voor fanout;
   - boot als eenmalige reflector;
   - eater1.

   Geen enkele timing komt uit het geheugen: alles is gesimuleerd. Opnieuw zoeken kan met `python -m gol.search`.
5. **Controle**:
   - `drc.py` zoekt geometrisch naar onbedoelde ontmoetingen.
   - `verify.py` simuleert en eist dat alleen de verwachte uitgangsgliders overblijven en dat het puin stil ligt.

## Verificatie

```bash
python -m pytest -q -m "not slow"    # snel: simulator, reacties, poorten, frontend
python -m pytest -q                  # alles, inclusief de trage programma-tests (~4 min)
```

- `tests/test_gates.py`: NOT, NOR2, NOR3, FAN en een poortketen, exhaustief in Life gesimuleerd.
- `tests/test_programs.py`: hele programma's in Life tegen Python.
  - Exhaustief: majority, xor3, 2-bit opteller, popcount.
  - Steekproef: max2 en de 4-bit opteller.
- `tests/test_frontend.py`: de Python-semantiek van frontend en techmap tegen echte Python, plus de foutmeldingen.
- `tests/test_reactions.py`: elke gebruikte reactie opnieuw gesimuleerd.
- `tests/test_life.py`: de simulator tegen een naïeve referentie, gliderbeweging, Gosper-gun met periode 30, RLE.

## Ondersteunende onderdelen: simulator en viewer

- `gol/life.py`: een snelle Life-simulator (bit-packed numpy, alleen actieve tegels, groeit automatisch).
- `gol/rle.py`: RLE laden en opslaan.
- `viewer/index.html` en `python -m gol.view x.rle`: een HTML/JS-viewer. Je kunt een RLE laden (bestand, drag-and-drop of plakken), afspelen, de snelheid kiezen, zoomen, pannen en naar een generatie springen.

```python
from gol import life, rle, patterns
u = life.Universe(patterns.GOSPER_GUN)
u.step(1000)
rle.save("gun.rle", u.cells)
```

```bash
python -m gol.view gun.rle          # schrijft gun.html (RLE ingebed) en opent hem
```

## Projectstructuur

```
gol/compiler/   de compiler: frontend, aig, techmap, netlist, layout, drc, verify, CLI
gol/editor.py   lokale webserver voor de editor
gol/            simulator, RLE, patronen, glider-hulpmiddelen, reactiezoektocht
editor/         editorpagina (HTML/JS)
data/           gevonden reacties (reactions.json)
examples/       voorbeeldprogramma's
tests/          pytest-suite
tools/          zoekscripts die bij het ontwerp gebruikt zijn
viewer/         HTML/JS-viewer
```

## Beperkingen

- **Eenmalig**: het patroon rekent één keer. Voor andere invoer maak je een nieuw beginpatroon (`prog.cells(...)`). Er zijn geen guns en geen klok.
- **Invoercodering**: een invoerbit bepaalt of één of meer gliders aanwezig zijn. Soms gaat het om gecomplementeerde gliders, die aanwezig zijn als de bit 0 is.
- **Grootte**: de patronen worden snel groot.
  - De 4-bit opteller is ongeveer 11 000 × 8 500 cellen en geeft zijn antwoord na ongeveer 34 000 generaties. Dat is zo'n 12 s simuleren.
  - Elke poort krijgt een eigen tijdslot, dus de duur groeit lineair met het aantal poorten.
- **Semantiek**: ints zijn unsigned en wrappen (zie `~` en `-` hierboven). Programma's die negatieve tussenresultaten vergelijken, geven dus een ander resultaat dan Python.
- **Uitvoer**: de uitkomst staat in gliders die wegvliegen. Er is geen blijvende display.

## Licentie

MIT, zie [LICENSE](LICENSE).
