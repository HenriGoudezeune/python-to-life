# Game of Life + Python→Life-compiler: ontwerpnotities

Persoonlijk project van Henri. Python 3.11+, numpy, scipy, pytest. Code en commentaar in het Engels, docs in het Nederlands.

## Structuur

| Pad | Inhoud |
|---|---|
| `gol/life.py` | Simulator: bit-packed numpy (64 cellen per uint64). `Universe` rekent alleen actieve tegels (32×64) door. |
| `gol/rle.py` | RLE lezen/schrijven. `#R x y` bewaart absolute coördinaten. `#C LABEL x y tekst` zijn labels voor de viewer. |
| `gol/patterns.py` | Glider, Gosper-gun, stillevens, symmetrieën. |
| `gol/gliders.py` | `Traj(dir, lane, tref)`: één exacte beschrijving van een glider voor alle tijden, plus `identify()`. |
| `gol/search.py` | Brute-force botsingszoektocht → `data/reactions.json` (`python -m gol.search`, ~1 min). |
| `gol/reactions.py` | Reactiebibliotheek uit de catalogus: transformeren (8 symmetrieën), verschuiven, plaatsen. |
| `gol/compiler/` | frontend (Python-AST → AIG), techmap (AIG → NOR/FAN-netlist), layout (netlist → gliders + stillevens), drc, verify, CLI. |
| `gol/editor.py`, `editor/index.html` | Editor-app: lokale stdlib-HTTP-server (127.0.0.1) met `/api/compile` en `/api/run`. De viewer draait in een iframe en krijgt het patroon via `loadText`. |
| `viewer/index.html` | Zelfstandige HTML/JS-viewer met tegel-simulator. `gol/view.py` bedt een RLE in. |
| `tools/search_fanout.py` | Zoektocht die de tweede constante van de FAN-poort vond. |
| `tests/` | pytest. `-m "not slow"` slaat de lange simulaties over. |

## Kernkeuze: one-shot gliderlogica, geen Gosper-gunstromen

Een signaal is één glider: aanwezig = 1, afwezig = 0. Alle constanten zijn losse gliders die bij generatie 0 al klaarstaan. Er zijn dus geen guns.
- Waarom: timing is exact en lokaal te berekenen. Kruisingen zijn eenvoudig (twee gliders botsen alleen als ze tegelijk op hetzelfde punt zijn). Er zijn geen periodieke interferenties. Bij een `0` blijven ongebruikte boten gewoon als statisch puin liggen.
- Nadeel: het patroon rekent één keer. Een nieuwe invoer vraagt een nieuw beginpatroon. Dat staat in de README bij de beperkingen.
- Invoer = de aanwezigheid van gliders op vaste posities. Een invoerbit mag meerdere gliders sturen, en ook gecomplementeerde gliders (aanwezig als de bit 0 is). Fanout van primaire invoer is dus gratis.

## Reacties (allemaal gevonden en geverifieerd door simulatie, niet uit het geheugen)

- Alle botsingen van twee gliders onder 90° en 180°, en van een glider met een stilleven, zijn gesimuleerd. Botsingen onder 90° hangen alleen af van Δθ en van de pariteit (lane_SE + lane_SW) mod 2. Bij |Δθ| ≥ 19 is er geen interactie.
- **K** (annihilatie): 22 schone varianten. **KB** (kickback): Δθ = 17 geeft een NE-glider terug.
- **Boot** = eenmalige 90°-reflector, schoon in alle fasen. **Eater1** eet een glider en overleeft. Eater1 kan ook als eenmalige reflector dienen.
- De tests in `tests/test_reactions.py` simuleren elke gebruikte reactie opnieuw.

## Timing: θ

- θ = t − 4y voor neerwaartse gliders (SE/SW) en t + 4y voor opwaartse (NE/NW). θ blijft constant langs een pad.
- Bij een reflectie verandert θ met een vaste constante.
- Een opwaarts stuk van d cellen geeft +8d.

## Layout ("bus")

- Een **rij** is een SE-glider op lane r = x − y en draagt een signaal. Een **kolom** is een SW-glider op lane c = x + y.
- Poort k krijgt kolom c_k (oplopend, stap `W`) en tijdslot θ_k (oplopend, stap `D`). De constante C_k komt van boven langs de kolom en kruist alle rijen.
  - Rijen voor poort k hebben precies de juiste Δθ en reageren.
  - Alle andere rijen hebben θ ≈ θ_h van hun eigen consument h ≠ k. Ze zitten dus minstens D − 15 ≥ 19 van θ_k af en vliegen erdoor.
- **NOR**: C_k sterft bij elke aanwezige invoerglider. Op de invoerrijen na de eerste staat een eater voor het geval dat C al dood was.
- **FAN** (één invoer → twee kopieën van NOT invoer):
  - C_k maakt een kickback met de invoerrij.
  - Een tweede constante G2 vernietigt de kickback-glider. Zonder invoer overleven C_k en G2 allebei.
  - G2 wordt via een kolom en een boot vlak links van c_k aangevoerd. Een glider van ver linksboven zou door oude vertragingslussen vliegen; dat was een echte bug.
- **Uitgang van een poort**: de overlevende constante loopt via een vertragingslus van 3 reflectoren (omlaag, rechts, omhoog, rechts) naar een **nieuwe rij onderaan de bus**.
  - De lengte van het opwaartse stuk zet θ exact op wat de consument nodig heeft.
  - Het opwaartse stuk kruist niets, omdat er onder de bus nog niets leeft.
- Lusvarianten (boot/eater1 per reflector) geven elke even θ-residu mod 8. Samen met de keuze uit de K-tabel en de kolompariteit is daardoor elke invoer haalbaar (zie `_input_target`).
- Uitgangen worden naar een laatste tijdslot vertraagd en vliegen rechts weg. Ze worden gelezen op `layout.t_end`.

## Verificatie

- `drc.check(layout)` controleert geometrisch elk paar padstukken op onbedoelde ontmoetingen, en elk stuk op onbedoelde stillevens in de buurt. Het is snel en vond de echte bugs.
- `verify.run` simuleert en controleert strikt: alleen de verwachte uitgangsgliders vliegen nog, en het overige puin is stil.
- Lessen: een stilleven mag nooit op het pad *stroomopwaarts* van een constante liggen. Constanten komen van ver: kolommen van boven, rijen van linksboven.

## Commando's

```bash
python -m pytest -q                      # alles (~1-3 min, opteller-tests zijn traag)
python -m pytest -q -m "not slow"        # snel
python -m gol.compiler examples/add4.py --inputs a=5,b=9 --run --drc --html
python -m gol.view out/add4_a5_b9.rle
python -m gol.editor                     # editor-app in de browser
python -m gol.search                     # catalogus opnieuw opbouwen
```

## Open / ideeën

- Kleinere patronen: θ-sloten hergebruiken, korter routeren, een XOR-primitief via K tussen twee datasignalen.
- HashLife voor nog grotere patronen.
- Een persistente uitvoer-"display": de uitgangsglider schrijft een stilleven.
