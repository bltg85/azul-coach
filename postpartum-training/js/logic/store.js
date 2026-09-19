// Lagring i localStorage. All data stannar på enheten.
const KEY = 'postpartum-training.v1';

export const emptyState = () => ({
  profile: null, // { birthDate, delivery, cleared, maxPhase, name, createdAt }
  sessions: [], // { id, date, phaseId, template, minutes, feeling, symptoms, note }
  kegelDays: {}, // { 'YYYY-MM-DD': antal gånger }
  runChecklist: {},
  settings: { reminders: false },
});

let memory = null;

export function load(storage = safeStorage()) {
  if (memory) return memory;
  try {
    const raw = storage && storage.getItem(KEY);
    memory = raw ? { ...emptyState(), ...JSON.parse(raw) } : emptyState();
  } catch {
    memory = emptyState();
  }
  return memory;
}

export function save(state, storage = safeStorage()) {
  memory = state;
  try {
    storage && storage.setItem(KEY, JSON.stringify(state));
  } catch {
    /* privat läge eller fullt – appen fungerar ändå under sessionen */
  }
  return state;
}

export function update(mutator) {
  const state = load();
  mutator(state);
  return save(state);
}

export function reset() {
  memory = emptyState();
  try {
    const s = safeStorage();
    s && s.removeItem(KEY);
  } catch {
    /* ignorera */
  }
  return memory;
}

export function exportJSON() {
  return JSON.stringify(load(), null, 2);
}

export function importJSON(text) {
  const parsed = JSON.parse(text);
  if (!parsed || typeof parsed !== 'object') throw new Error('Ogiltig fil');
  return save({ ...emptyState(), ...parsed });
}

function safeStorage() {
  try {
    return typeof localStorage !== 'undefined' ? localStorage : null;
  } catch {
    return null;
  }
}
