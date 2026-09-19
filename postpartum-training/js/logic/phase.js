// Ren logik för faser och tidslinje. Inga DOM-beroenden – testas med node --test.

export const PHASES = [
  {
    id: 'p0',
    index: 0,
    name: 'Vila & andning',
    short: 'Fas 1',
    startWeek: 0,
    csOffset: 0,
    requiresClearance: false,
    walkMinutes: [5, 15],
    color: 'var(--phase0)',
    summary:
      'Kroppen läker. Fokus på andning, mjuk kontakt med bäckenbotten och korta promenader när det känns bra.',
    goals: [
      'Andas djupt och lugnt flera gånger om dagen',
      'Hitta kontakten med bäckenbotten – både knip och avslappning',
      'Korta promenader, gärna flera små i stället för en lång',
      'Vila. Sömn och återhämtning är träning nu',
    ],
    avoid: ['Tunga lyft (tyngre än bebisen)', 'Situps, crunches och planka', 'Löpning och hopp'],
  },
  {
    id: 'p1',
    index: 1,
    name: 'Återhämtning',
    short: 'Fas 2',
    startWeek: 2,
    csOffset: 1,
    requiresClearance: false,
    walkMinutes: [15, 30],
    color: 'var(--phase1)',
    summary:
      'Bygg upp de djupa musklerna kring bäcken och mage med lugna, kontrollerade övningar.',
    goals: [
      'Bäckenbottenträning varje dag',
      'Aktivera de djupa magmusklerna utan att magen buktar',
      'Lätta övningar för säte och höfter',
      'Promenader 15–30 minuter',
    ],
    avoid: ['Situps och crunches', 'Tunga lyft', 'Löpning, hopp och intensiva pass'],
  },
  {
    id: 'p2',
    index: 2,
    name: 'Grund',
    short: 'Fas 3',
    startWeek: 6,
    csOffset: 0,
    requiresClearance: true,
    walkMinutes: [30, 45],
    color: 'var(--phase2)',
    summary:
      'Efter efterkontrollen: styrka med kroppsvikt och gummiband, fortsatt fokus på bäckenbotten och bål.',
    goals: [
      'Styrkepass 2–3 gånger i veckan',
      'Knäböj, step-ups och rodd med god kontroll',
      'Bålstabilitet: fågelhund, sidoplanka på knä, dead bug',
      'Raska promenader eller lätt cykling',
    ],
    avoid: ['Övningar där magen buktar (doming)', 'Hopp och löpning om du läcker eller känner tyngd'],
  },
  {
    id: 'p3',
    index: 3,
    name: 'Uppbyggnad',
    short: 'Fas 4',
    startWeek: 12,
    csOffset: 2,
    requiresClearance: true,
    walkMinutes: [30, 60],
    color: 'var(--phase3)',
    summary:
      'Öka belastningen stegvis. Om du klarar checklistan kan du börja gå/jogga-intervaller.',
    goals: [
      'Styrkepass 2–3 gånger i veckan med mer belastning',
      'Planka och sidoplanka utan att magen buktar',
      'Gå/jogga-intervaller om checklistan är klar',
      'Hitta tillbaka till aktiviteter du gillar – i din takt',
    ],
    avoid: ['Att öka allt på en gång – ändra en sak i taget', 'Att ignorera läckage, tyngd eller smärta'],
  },
];

const MS_PER_DAY = 24 * 60 * 60 * 1000;

/** Antal hela veckor sedan förlossningen (aldrig negativt). */
export function weeksSince(birthDateISO, today = new Date()) {
  const birth = parseISODate(birthDateISO);
  const now = startOfDay(today);
  const days = Math.floor((now - birth) / MS_PER_DAY);
  return Math.max(0, Math.floor(days / 7));
}

export function daysSince(birthDateISO, today = new Date()) {
  const birth = parseISODate(birthDateISO);
  const now = startOfDay(today);
  return Math.max(0, Math.floor((now - birth) / MS_PER_DAY));
}

/** Startvecka för en fas givet förlossningssätt. */
export function phaseStartWeek(phase, delivery) {
  return phase.startWeek + (delivery === 'kejsarsnitt' ? phase.csOffset : 0);
}

/**
 * Räknar ut aktuell fas.
 * @param {object} profile { birthDate, delivery: 'vaginal'|'kejsarsnitt', cleared: boolean, maxPhase?: number }
 * @returns {{ phase, week, weekInPhase, capped: null|'clearance'|'manual', nextPhase, weeksToNext }}
 */
export function currentPhase(profile, today = new Date()) {
  const week = weeksSince(profile.birthDate, today);
  const delivery = profile.delivery || 'vaginal';

  let eligible = PHASES[0];
  let capped = null;
  for (const p of PHASES) {
    if (week < phaseStartWeek(p, delivery)) break;
    if (p.requiresClearance && !profile.cleared) {
      capped = 'clearance';
      break;
    }
    eligible = p;
  }

  const maxPhase = Number.isInteger(profile.maxPhase) ? profile.maxPhase : PHASES.length - 1;
  if (eligible.index > maxPhase) {
    eligible = PHASES[Math.max(0, maxPhase)];
    capped = 'manual';
  }

  const weekInPhase = Math.max(0, week - phaseStartWeek(eligible, delivery));
  const nextPhase = PHASES[eligible.index + 1] || null;
  const weeksToNext = nextPhase ? Math.max(0, phaseStartWeek(nextPhase, delivery) - week) : null;

  return { phase: eligible, week, weekInPhase, capped, nextPhase, weeksToNext };
}

/** Varningssignaler som betyder: pausa och kontakta vården. */
export const RED_FLAGS = [
  { id: 'bleeding', label: 'Blödningen ökar eller blir klarröd igen' },
  { id: 'fever', label: 'Feber eller frossa' },
  { id: 'wound', label: 'Sår som rodnar, svullnar eller vätskar' },
  { id: 'pain', label: 'Smärta som blir värre, i mage, bäcken eller ärr' },
  { id: 'heaviness', label: 'Tyngdkänsla eller något som buktar ut i underlivet' },
  { id: 'leak', label: 'Läckage av urin eller avföring som blir värre' },
  { id: 'calf', label: 'Svullen, öm eller varm vad' },
  { id: 'breath', label: 'Andnöd eller bröstsmärta' },
];

/** Symtom som kan loggas efter ett pass. */
export const SYMPTOMS = [
  { id: 'leak', label: 'Läckage', advice: 'Backa till lugnare övningar och prioritera bäckenbottenträning. Kontakta fysioterapeut om det fortsätter.' },
  { id: 'heaviness', label: 'Tyngdkänsla', advice: 'Undvik stående belastning och hopp ett tag. Träna liggande. Kontakta fysioterapeut om det fortsätter.' },
  { id: 'doming', label: 'Magen buktade', advice: 'Gör övningen lättare eller byt till en annan. Andas ut och aktivera djupa magmuskler före varje rörelse.' },
  { id: 'pain', label: 'Smärta', advice: 'Smärta är en signal att stanna. Backa ett steg, och kontakta vården om smärtan inte går över.' },
  { id: 'tired', label: 'Väldigt trött', advice: 'Vila är också träning. Ta ett kortare pass eller en promenad nästa gång.' },
];

/**
 * Ger råd utifrån de senaste passens symtom.
 * @returns {{ level: 'ok'|'ease'|'contact', symptomsIds: string[], message: string }}
 */
export function evaluateSymptoms(recentSessions) {
  const counts = {};
  for (const s of recentSessions) {
    for (const id of s.symptoms || []) counts[id] = (counts[id] || 0) + 1;
  }
  const flagged = Object.keys(counts).filter((id) => ['leak', 'heaviness', 'pain', 'doming'].includes(id));
  if (flagged.length === 0) return { level: 'ok', symptomIds: [], message: 'Inga symtom loggade på sistone. Fortsätt i din takt.' };
  const persistent = flagged.filter((id) => counts[id] >= 2);
  if (persistent.length > 0) {
    return {
      level: 'contact',
      symptomIds: persistent,
      message:
        'Du har haft samma symtom vid flera pass. Backa ett steg i programmet och kontakta barnmorska eller fysioterapeut.',
    };
  }
  return {
    level: 'ease',
    symptomIds: flagged,
    message: 'Du loggade symtom senast. Ta ett lugnare pass i dag och känn efter.',
  };
}

/** Checklista innan gå/jogga-intervaller (fas 4). */
export const RUN_CHECKLIST = [
  { id: 'weeks', label: 'Minst 12 veckor har gått sedan förlossningen' },
  { id: 'walk30', label: 'Jag kan gå raskt i 30 minuter utan besvär' },
  { id: 'hop', label: 'Jag kan hoppa på stället 10 gånger utan läckage eller tyngd' },
  { id: 'singleleg', label: 'Jag kan stå på ett ben i 10 sekunder per sida med kontroll' },
  { id: 'squat', label: 'Jag kan göra 20 knäböj och 10 enbensknäböj per sida utan smärta' },
  { id: 'nosymptoms', label: 'Inget läckage, ingen tyngdkänsla och ingen smärta i vardagen' },
];

export function runReady(checked = {}) {
  return RUN_CHECKLIST.every((c) => checked[c.id]);
}

export function parseISODate(iso) {
  const [y, m, d] = String(iso).split('-').map(Number);
  return new Date(y, (m || 1) - 1, d || 1);
}

export function startOfDay(date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

export function toISODate(date = new Date()) {
  const d = startOfDay(date);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${d.getFullYear()}-${mm}-${dd}`;
}
