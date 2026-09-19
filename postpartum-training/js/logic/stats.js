// Statistik över loggade pass. Ren logik.
import { toISODate, parseISODate, startOfDay } from './phase.js';

const MS_PER_DAY = 24 * 60 * 60 * 1000;

export function sessionsInLastDays(sessions, days, today = new Date()) {
  const end = startOfDay(today).getTime();
  const start = end - (days - 1) * MS_PER_DAY;
  return sessions.filter((s) => {
    const t = parseISODate(s.date).getTime();
    return t >= start && t <= end;
  });
}

/** Antal dagar i rad (t.o.m. i dag eller i går) med minst en aktivitet. */
export function streak(sessions, kegelDays = {}, today = new Date()) {
  const active = new Set(sessions.map((s) => s.date));
  for (const [d, n] of Object.entries(kegelDays)) if (n > 0) active.add(d);
  let day = startOfDay(today);
  if (!active.has(toISODate(day))) day = new Date(day.getTime() - MS_PER_DAY);
  let count = 0;
  while (active.has(toISODate(day))) {
    count += 1;
    day = new Date(day.getTime() - MS_PER_DAY);
  }
  return count;
}

/** Senaste 7 dagarna som lista: { date, sessions, kegel } – äldst först. */
export function lastWeek(sessions, kegelDays = {}, today = new Date()) {
  const out = [];
  const end = startOfDay(today).getTime();
  for (let i = 6; i >= 0; i -= 1) {
    const d = new Date(end - i * MS_PER_DAY);
    const iso = toISODate(d);
    out.push({
      date: iso,
      weekday: d.toLocaleDateString('sv-SE', { weekday: 'short' }),
      sessions: sessions.filter((s) => s.date === iso),
      kegel: kegelDays[iso] || 0,
    });
  }
  return out;
}

export function totalMinutes(sessions) {
  return sessions.reduce((sum, s) => sum + (Number(s.minutes) || 0), 0);
}
