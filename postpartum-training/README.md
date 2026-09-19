# Igång igen – träning efter förlossningen

En liten, trygg webbapp (PWA) för att komma igång med träningen efter förlossningen.
Allt på svenska, inget konto, ingen server – all data stannar i din telefon.

## Vad appen gör

- **Fyra faser** som följer kroppens läkning, räknat från förlossningsdatumet:
  1. *Vila & andning* (vecka 0–2): andning, kontakt med bäckenbotten, korta promenader
  2. *Återhämtning* (vecka 2–6): bäckenbotten, djupa magmuskler, säte och höfter
  3. *Grund* (vecka 6–12, kräver klartecken från efterkontrollen): styrka med kroppsvikt och gummiband
  4. *Uppbyggnad* (vecka 12+): mer belastning, planka, och gå/jogga-intervaller när checklistan är klar
- **Kejsarsnitt** förskjuter fas 2 och fas 4 automatiskt.
- **Dagens pass** i två längder (kort ≈ 10 min, standard ≈ 20 min) med guidad passpelare, timer, set-räknare och instruktioner för varje övning. Dosen ökar per vecka i fasen.
- **Promenad** och **gå/jogga-progression** (8 veckor) med checklista innan löpning.
- **Bäckenbotten-räknare**: tre omgångar knip om dagen med en knapptryckning.
- **Logg** med känsla, symtom (läckage, tyngdkänsla, smärta, doming) och anteckning. Upprepade symtom ger råd att backa och kontakta vården.
- **Lär dig**: bäckenbotten, magmuskeldelning, kejsarsnitt, varningssignaler, löpning, vardag och mående.
- **Ta det lugnare**: möjlighet att låsa programmet i en lägre fas.
- Export/import av data som JSON, mörkt läge, fungerar offline (service worker) och kan läggas på hemskärmen.

> Appen ger allmän information och ersätter inte råd från barnmorska, läkare eller fysioterapeut.

## Kör lokalt

Ingen byggprocess. Servera mappen statiskt:

```bash
cd postpartum-training
npm start            # python3 -m http.server 8080
# öppna http://localhost:8080
```

Valfri statisk server fungerar (`npx serve`, GitHub Pages, Netlify, Vercel).

## Tester

Logiken (faser, dosering, symtombedömning, statistik) är rena moduler utan DOM och testas med Node:

```bash
npm test             # node --test
```

## Struktur

```
index.html            Appskal
css/style.css         Stil, ljust/mörkt läge
js/app.js             Vyer, router, passpelare
js/ui.js              HTML-mall med escaping, toast, formatering
js/logic/phase.js     Faser, tidslinje, varningssignaler, symtombedömning, löpchecklista
js/logic/plan.js      Bygger pass från fas + vecka
js/logic/store.js     localStorage
js/logic/stats.js     Streak och veckostatistik
js/data/exercises.js  Övningsbibliotek (25 övningar)
js/data/program.js    Passmallar per fas, löpprogression
js/data/learn.js      Utbildningstexter
tests/                node --test
sw.js, manifest.webmanifest, icons/   PWA
```

## Flytta till eget repo

Mappen är fristående. Kopiera den till ett nytt repo, eller:

```bash
git subtree split -P postpartum-training -b postpartum-only
```
