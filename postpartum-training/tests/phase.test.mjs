import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  weeksSince,
  currentPhase,
  evaluateSymptoms,
  runReady,
  RUN_CHECKLIST,
  toISODate,
} from '../js/logic/phase.js';
import { buildSession, estimateMinutes, describeDose } from '../js/logic/plan.js';
import { streak, sessionsInLastDays } from '../js/logic/stats.js';
import { PROGRAM } from '../js/data/program.js';
import { EXERCISES } from '../js/data/exercises.js';

const today = new Date(2026, 8, 19); // 19 sep 2026
const birthWeeksAgo = (w) => {
  const d = new Date(today.getTime() - w * 7 * 24 * 3600 * 1000);
  return toISODate(d);
};

test('weeksSince räknar hela veckor och aldrig negativt', () => {
  assert.equal(weeksSince(birthWeeksAgo(0), today), 0);
  assert.equal(weeksSince(birthWeeksAgo(3), today), 3);
  assert.equal(weeksSince('2026-12-01', today), 0);
});

test('vaginal förlossning: faser efter vecka och efterkontroll', () => {
  const p = (w, cleared = false, maxPhase) =>
    currentPhase({ birthDate: birthWeeksAgo(w), delivery: 'vaginal', cleared, maxPhase }, today);
  assert.equal(p(0).phase.id, 'p0');
  assert.equal(p(1).phase.id, 'p0');
  assert.equal(p(2).phase.id, 'p1');
  assert.equal(p(5).phase.id, 'p1');
  const uncleared = p(7);
  assert.equal(uncleared.phase.id, 'p1');
  assert.equal(uncleared.capped, 'clearance');
  assert.equal(p(7, true).phase.id, 'p2');
  assert.equal(p(12, true).phase.id, 'p3');
  assert.equal(p(12, true).nextPhase, null);
  assert.equal(p(2).weeksToNext, 4);
});

test('kejsarsnitt skjuter fram fas 2 och fas 4', () => {
  const p = (w, cleared = true) =>
    currentPhase({ birthDate: birthWeeksAgo(w), delivery: 'kejsarsnitt', cleared }, today);
  assert.equal(p(2).phase.id, 'p0');
  assert.equal(p(3).phase.id, 'p1');
  assert.equal(p(6).phase.id, 'p2');
  assert.equal(p(13).phase.id, 'p2');
  assert.equal(p(14).phase.id, 'p3');
  assert.equal(p(3).weekInPhase, 0);
  assert.equal(p(5).weekInPhase, 2);
});

test('manuellt tak på fas respekteras', () => {
  const r = currentPhase({ birthDate: birthWeeksAgo(20), delivery: 'vaginal', cleared: true, maxPhase: 1 }, today);
  assert.equal(r.phase.id, 'p1');
  assert.equal(r.capped, 'manual');
});

test('alla mallar refererar befintliga övningar och ger rimlig tid', () => {
  for (const [phaseId, templates] of Object.entries(PROGRAM)) {
    for (const template of Object.keys(templates)) {
      const s = buildSession(phaseId, template, 0);
      assert.ok(s.length >= 5, `${phaseId}/${template}`);
      for (const ex of s) assert.ok(EXERCISES[ex.id]);
      const min = estimateMinutes(s);
      assert.ok(min >= 4 && min <= 35, `${phaseId}/${template}: ${min} min`);
    }
  }
});

test('dosen ökar per vecka men stannar vid max', () => {
  const w0 = buildSession('p1', 'kort', 0).find((e) => e.id === 'gluteBridge');
  const w2 = buildSession('p1', 'kort', 2).find((e) => e.id === 'gluteBridge');
  const w9 = buildSession('p1', 'kort', 9).find((e) => e.id === 'gluteBridge');
  assert.equal(w0.amount, 10);
  assert.equal(w2.amount, 14);
  assert.equal(w9.amount, 15);
  assert.equal(describeDose(w0), '2 × 10');
  assert.throws(() => buildSession('p9', 'kort'));
});

test('symtombedömning: upprepade symtom ger kontakt-nivå', () => {
  assert.equal(evaluateSymptoms([]).level, 'ok');
  assert.equal(evaluateSymptoms([{ symptoms: ['tired'] }]).level, 'ok');
  assert.equal(evaluateSymptoms([{ symptoms: ['leak'] }]).level, 'ease');
  const r = evaluateSymptoms([{ symptoms: ['leak'] }, { symptoms: ['leak', 'pain'] }]);
  assert.equal(r.level, 'contact');
  assert.deepEqual(r.symptomIds, ['leak']);
});

test('löpchecklista kräver alla punkter', () => {
  assert.equal(runReady({}), false);
  const all = Object.fromEntries(RUN_CHECKLIST.map((c) => [c.id, true]));
  assert.equal(runReady(all), true);
  assert.equal(runReady({ ...all, hop: false }), false);
});

test('streak räknar dagar i rad inklusive knipdagar', () => {
  const d = (i) => toISODate(new Date(today.getTime() - i * 24 * 3600 * 1000));
  assert.equal(streak([], {}, today), 0);
  assert.equal(streak([{ date: d(0) }, { date: d(1) }], {}, today), 2);
  assert.equal(streak([{ date: d(1) }], { [d(2)]: 1 }, today), 2);
  assert.equal(streak([{ date: d(1) }, { date: d(3) }], {}, today), 1);
  assert.equal(sessionsInLastDays([{ date: d(0) }, { date: d(6) }, { date: d(7) }], 7, today).length, 2);
});
