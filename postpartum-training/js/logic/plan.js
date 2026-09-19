// Bygger dagens pass från fas, vecka i fasen och mall. Ren logik.
import { EXERCISES } from '../data/exercises.js';
import { PROGRAM } from '../data/program.js';

export function dose(exercise, weekInPhase) {
  const { base, step, max } = exercise.dose;
  return Math.min(max, base + step * weekInPhase);
}

/**
 * @returns {Array<{id, name, sets, amount, type, perSide, hold, cue, why, caution, category}>}
 */
export function buildSession(phaseId, template, weekInPhase = 0) {
  const rows = (PROGRAM[phaseId] || {})[template];
  if (!rows) throw new Error(`Okänd mall: ${phaseId}/${template}`);
  return rows.map(([id, sets]) => {
    const ex = EXERCISES[id];
    if (!ex) throw new Error(`Okänd övning: ${id}`);
    return {
      id,
      name: ex.name,
      category: ex.category,
      sets,
      amount: dose(ex, weekInPhase),
      type: ex.type,
      perSide: !!ex.perSide,
      hold: ex.hold || null,
      cue: ex.cue,
      why: ex.why,
      caution: ex.caution || null,
    };
  });
}

/** Ungefärlig tid i minuter för ett pass. */
export function estimateMinutes(session) {
  let seconds = 0;
  for (const ex of session) {
    const sides = ex.perSide ? 2 : 1;
    const perSet = ex.type === 'time' ? ex.amount : ex.amount * (ex.hold ? ex.hold + 2 : 3);
    seconds += ex.sets * sides * perSet + ex.sets * 20; // 20 s vila/set
  }
  return Math.max(1, Math.round(seconds / 60));
}

export function describeDose(ex) {
  const side = ex.perSide ? '/sida' : '';
  if (ex.type === 'time') return `${ex.sets} × ${ex.amount} s${side}`;
  const hold = ex.hold ? ` à ${ex.hold} s` : '';
  return `${ex.sets} × ${ex.amount}${side}${hold}`;
}
