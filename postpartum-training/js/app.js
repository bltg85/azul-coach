import { html, raw, toast, formatDate, formatSeconds, uid } from './ui.js';
import {
  PHASES, currentPhase, daysSince, phaseStartWeek, RED_FLAGS, SYMPTOMS, evaluateSymptoms,
  RUN_CHECKLIST, runReady, toISODate,
} from './logic/phase.js';
import { buildSession, estimateMinutes, describeDose } from './logic/plan.js';
import { load, update, reset, exportJSON, importJSON } from './logic/store.js';
import { streak, lastWeek, sessionsInLastDays, totalMinutes } from './logic/stats.js';
import { EXERCISES, CATEGORY_LABELS } from './data/exercises.js';
import { PROGRAM, TEMPLATE_LABELS, RUN_PROGRESSION } from './data/program.js';
import { LEARN } from './data/learn.js';

const FEELINGS = [
  { id: 'bra', em: '😊', label: 'Bra' },
  { id: 'okej', em: '😐', label: 'Okej' },
  { id: 'jobbigt', em: '😣', label: 'Jobbigt' },
];
const NAV = [
  { id: 'home', label: 'Idag', icon: '🏠' },
  { id: 'program', label: 'Program', icon: '📋' },
  { id: 'log', label: 'Logg', icon: '📈' },
  { id: 'learn', label: 'Lär dig', icon: '📖' },
  { id: 'more', label: 'Mer', icon: '⚙️' },
];
const KEGEL_GOAL = 3;

// ---------- Router ----------

function parseRoute() {
  const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  return { name: parts[0] || 'home', a: parts[1], b: parts[2] };
}

function go(path) {
  location.hash = `#/${path}`;
}

function render() {
  const state = load();
  const route = parseRoute();
  const view = document.getElementById('view');
  if (!state.profile && route.name !== 'onboarding') return go('onboarding');
  if (state.profile && route.name === 'onboarding') return go('home');

  const views = {
    onboarding: viewOnboarding,
    home: viewHome,
    session: viewSession,
    walk: viewWalk,
    run: viewRun,
    program: viewProgram,
    log: viewLog,
    learn: viewLearn,
    more: viewMore,
  };
  const fn = views[route.name] || viewHome;
  if (route.name !== 'session') stopTimer();
  view.innerHTML = fn(state, route);
  renderNav(route.name, !!state.profile);
  renderTopbar(state);
  afterRender(route, state);
  window.scrollTo({ top: 0 });
}

function renderNav(active, show) {
  const nav = document.getElementById('nav');
  nav.innerHTML = show
    ? NAV.map(
        (n) => html`<a href="#/${n.id}" class="${n.id === active ? 'active' : ''}" aria-current="${n.id === active ? 'page' : 'false'}"><span class="i" aria-hidden="true">${n.icon}</span>${n.label}</a>`
      ).join('')
    : '';
}

function renderTopbar(state) {
  const el = document.getElementById('topbar-meta');
  if (!state.profile) return (el.textContent = '');
  const { week } = currentPhase(state.profile);
  const days = daysSince(state.profile.birthDate);
  el.textContent = `Vecka ${week} · dag ${days}`;
}

// ---------- Onboarding ----------

function viewOnboarding() {
  const today = toISODate();
  return html`
    <div class="stack">
      <div class="card center">
        <div class="hero-emoji">🌱</div>
        <h1>Välkommen</h1>
        <p>Ett tryggt sätt att komma igång med träningen efter förlossningen – i din egen takt, fem minuter i taget.</p>
      </div>
      <form class="card" id="onboarding">
        <label class="field"><span>Vad vill du bli kallad? <span class="muted small">(frivilligt)</span></span>
          <input type="text" name="name" maxlength="30" autocomplete="given-name" /></label>
        <label class="field"><span>Förlossningsdatum</span>
          <input type="date" name="birthDate" required max="${today}" value="${today}" />
          <span class="hint">Programmet räknar veckor från det här datumet.</span></label>
        <div class="field"><span>Förlossningssätt</span>
          <div class="choice">
            <label><input type="radio" name="delivery" value="vaginal" checked /> <span><strong>Vaginal förlossning</strong></span></label>
            <label><input type="radio" name="delivery" value="kejsarsnitt" /> <span><strong>Kejsarsnitt</strong><br /><span class="muted small">Programmet tar det lite lugnare med bål och belastning.</span></span></label>
          </div>
        </div>
        <label class="check"><input type="checkbox" name="cleared" /> <span>Jag har gjort efterkontrollen och fått klartecken att träna<br /><span class="muted small">Krävs för fas 3 och 4. Du kan kryssa i senare under Mer.</span></span></label>
        <div class="banner warn" style="margin:12px 0">
          <span class="icon">🚩</span>
          <div><strong>Sök vård om du får</strong> ökad blödning, feber, sårproblem, tilltagande smärta, tyngdkänsla i underlivet, svullen vad eller andnöd. Appen ersätter inte råd från barnmorska, läkare eller fysioterapeut.</div>
        </div>
        <label class="check"><input type="checkbox" name="ack" required /> <span>Jag förstår att appen ger allmän information och att jag lyssnar på min kropp.</span></label>
        <button class="btn primary block" type="submit" style="margin-top:12px">Kom igång</button>
      </form>
    </div>`;
}

// ---------- Hem ----------

function viewHome(state) {
  const p = state.profile;
  const info = currentPhase(p);
  const { phase, week, weekInPhase, capped, nextPhase, weeksToNext } = info;
  const today = toISODate();
  const doneToday = state.sessions.filter((s) => s.date === today);
  const kegelToday = state.kegelDays[today] || 0;
  const last3 = state.sessions.slice(-3);
  const evalu = evaluateSymptoms(last3);
  const weekSessions = sessionsInLastDays(state.sessions, 7);
  const span = nextPhase ? phaseStartWeek(nextPhase, p.delivery) - phaseStartWeek(phase, p.delivery) : 8;
  const pct = Math.min(100, Math.round(((weekInPhase + 1) / span) * 100));
  const standardMin = estimateMinutes(buildSession(phase.id, 'standard', weekInPhase));
  const kortMin = estimateMinutes(buildSession(phase.id, 'kort', weekInPhase));
  const runOk = phase.id === 'p3' && runReady(state.runChecklist);
  const hour = new Date().getHours();
  const greet = hour < 10 ? 'God morgon' : hour < 18 ? 'Hej' : 'God kväll';

  return html`
    <div class="stack">
      <h1>${greet}${p.name ? `, ${p.name}` : ''}!</h1>

      <section class="phase-hero" style="background:${phase.color}">
        <div class="kicker">${phase.short} av 4 · vecka ${week} efter förlossningen</div>
        <h2>${phase.name}</h2>
        <p>${phase.summary}</p>
        <div class="progress" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><span style="width:${pct}%"></span></div>
        <p class="small" style="margin-top:6px;opacity:.85">
          ${nextPhase
            ? weeksToNext === 0
              ? capped === 'clearance'
                ? `Nästa fas väntar på klartecken från efterkontrollen.`
                : `Redo för nästa fas – när du känner dig redo.`
              : `Nästa fas (${nextPhase.name}) om ${weeksToNext} ${weeksToNext === 1 ? 'vecka' : 'veckor'}.`
            : 'Sista fasen – nu handlar det om att bygga vidare i din takt.'}
        </p>
      </section>

      ${capped === 'clearance'
        ? html`<div class="banner warn"><span class="icon">🩺</span><div><strong>Dags för efterkontroll?</strong> Fas 3 låses upp när du fått klartecken från barnmorska eller läkare.<div style="margin-top:8px"><button class="btn sm" data-action="set-cleared">Jag har fått klartecken</button></div></div></div>`
        : ''}
      ${capped === 'manual'
        ? html`<div class="banner info"><span class="icon">🐢</span><div>Du har valt att stanna i ${phase.name}. Ändra under <a href="#/more">Mer</a> när du vill gå vidare.</div></div>`
        : ''}
      ${evalu.level !== 'ok'
        ? html`<div class="banner ${evalu.level === 'contact' ? 'danger' : 'warn'}"><span class="icon">${evalu.level === 'contact' ? '📞' : '💡'}</span><div>${evalu.message}</div></div>`
        : ''}

      <section class="card">
        <div class="card-head"><h2>Dagens pass</h2>${doneToday.length ? html`<span class="chip ok"><span class="dot"></span>${doneToday.length} klart i dag</span>` : ''}</div>
        <div class="options">
          <button class="option" data-action="start" data-template="kort"><span class="em">⚡</span><span><strong>Kort pass · ca ${kortMin} min</strong><span class="muted">${TEMPLATE_LABELS.kort.blurb}</span></span><span class="arrow">›</span></button>
          <button class="option" data-action="start" data-template="standard"><span class="em">🧘‍♀️</span><span><strong>Standardpass · ca ${standardMin} min</strong><span class="muted">${TEMPLATE_LABELS.standard.blurb}</span></span><span class="arrow">›</span></button>
          <a class="option" href="#/walk"><span class="em">🚶‍♀️</span><span><strong>Promenad · ${phase.walkMinutes[0]}–${phase.walkMinutes[1]} min</strong><span class="muted">${TEMPLATE_LABELS.promenad.blurb}</span></span><span class="arrow">›</span></a>
          ${phase.id === 'p3'
            ? runOk
              ? html`<a class="option" href="#/run"><span class="em">🏃‍♀️</span><span><strong>Gå/jogga-intervaller</strong><span class="muted">Vecka ${runWeekIndex(state) + 1} av 8 i löpprogressionen.</span></span><span class="arrow">›</span></a>`
              : html`<a class="option" href="#/learn/lopning"><span class="em">🏃‍♀️</span><span><strong>Vill du börja jogga?</strong><span class="muted">Gå igenom checklistan först.</span></span><span class="arrow">›</span></a>`
            : ''}
        </div>
      </section>

      <section class="card">
        <div class="card-head"><h2>Bäckenbotten i dag</h2><span class="chip ${kegelToday >= KEGEL_GOAL ? 'ok' : ''}">${kegelToday} / ${KEGEL_GOAL}</span></div>
        <p class="muted small">Tre omgångar knip om dagen – när du ammar, väntar på kaffet eller ligger i sängen. 8–10 långsamma (håll 5 s) och 10 snabba.</p>
        <div class="btn-row"><button class="btn" data-action="kegel">+ Knip gjort</button>${kegelToday > 0 ? html`<button class="btn ghost sm" data-action="kegel-undo">Ångra</button>` : ''}</div>
      </section>

      <section class="stats">
        <div class="stat"><div class="n">${streak(state.sessions, state.kegelDays)}</div><div class="l">dagar i rad</div></div>
        <div class="stat"><div class="n">${weekSessions.length}</div><div class="l">pass senaste 7 dagar</div></div>
        <div class="stat"><div class="n">${totalMinutes(weekSessions)}</div><div class="l">minuter senaste 7 dagar</div></div>
      </section>
    </div>`;
}

// ---------- Passpelare ----------

let player = null;
let timerId = null;

function stopTimer() {
  if (timerId) clearInterval(timerId);
  timerId = null;
  if (player) player.running = false;
}

function initPlayer(state, template) {
  const { phase, weekInPhase } = currentPhase(state.profile);
  const session = buildSession(phase.id, template, weekInPhase);
  player = { template, phaseId: phase.id, session, idx: 0, set: 1, side: 1, remaining: session[0].amount, running: false, startedAt: Date.now(), finished: false };
}

function viewSession(state, route) {
  const template = route.a;
  if (!PROGRAM[currentPhase(state.profile).phase.id][template]) return go('home');
  if (!player || player.template !== template || player.finished === 'saved') initPlayer(state, template);
  if (player.finished) return viewFeedback(state, { template, minutes: Math.max(1, Math.round((Date.now() - player.startedAt) / 60000)) });

  const ex = player.session[player.idx];
  const n = player.session.length;
  const pct = Math.round((player.idx / n) * 100);
  const sets = Array.from({ length: ex.sets }, (_, i) => html`<span class="${i + 1 < player.set || (i + 1 === player.set && ex.perSide && player.side === 2) ? 'done' : ''}"></span>`);

  return html`
    <div class="player">
      <div class="player-top">
        <span>Övning ${player.idx + 1} av ${n}</span>
        <a href="#/home" class="btn ghost sm" data-action="abort">Avsluta</a>
      </div>
      <div class="player-bar"><span style="width:${pct}%"></span></div>
      <section class="card">
        <span class="chip">${CATEGORY_LABELS[ex.category]}</span>
        <h2 style="margin-top:8px">${ex.name}</h2>
        <p class="muted">${describeDose(ex)}</p>
        ${ex.caution ? html`<div class="banner warn small"><span class="icon">⚠️</span><div>${ex.caution}</div></div>` : ''}
        ${ex.type === 'time'
          ? html`<div class="timer"><div class="t" id="timer-text">${formatSeconds(player.remaining)}</div>${ex.perSide ? html`<div class="side">Sida ${player.side} av 2</div>` : ''}</div>`
          : html`<div class="timer"><div class="big-count">${ex.amount}${ex.hold ? html`<span class="muted" style="font-size:1rem"> × håll ${ex.hold} s</span>` : ''}</div>${ex.perSide ? html`<div class="side">Sida ${player.side} av 2</div>` : ''}</div>`}
        <div class="sets" aria-label="Set">${sets}</div>
        <p class="center muted small">Set ${player.set} av ${ex.sets}</p>
        <div class="btn-row" style="margin-top:8px">
          ${ex.type === 'time'
            ? html`<button class="btn primary" data-action="toggle-timer" id="timer-btn">${player.running ? 'Pausa' : player.remaining < ex.amount ? 'Fortsätt' : 'Starta'}</button><button class="btn" data-action="complete-set">Klar med set</button>`
            : html`<button class="btn primary" data-action="complete-set">Set klart</button>`}
        </div>
      </section>
      <section class="card soft">
        <h3>Så gör du</h3>
        <p>${ex.cue}</p>
        <details><summary>Varför den här?</summary><div class="body">${ex.why}</div></details>
      </section>
      <div class="btn-row">
        <button class="btn ghost" data-action="prev" ${player.idx === 0 ? 'disabled' : ''}>‹ Föregående</button>
        <button class="btn ghost" data-action="next">Hoppa över ›</button>
      </div>
    </div>`;
}

function completeSet() {
  const ex = player.session[player.idx];
  stopTimer();
  if (ex.perSide && player.side === 1) {
    player.side = 2;
  } else if (player.set < ex.sets) {
    player.set += 1;
    player.side = 1;
  } else {
    return nextExercise();
  }
  player.remaining = ex.amount;
  render();
}

function nextExercise() {
  stopTimer();
  if (player.idx + 1 >= player.session.length) {
    player.finished = true;
  } else {
    player.idx += 1;
    player.set = 1;
    player.side = 1;
    player.remaining = player.session[player.idx].amount;
  }
  render();
}

function prevExercise() {
  stopTimer();
  if (player.idx === 0) return;
  player.idx -= 1;
  player.set = 1;
  player.side = 1;
  player.remaining = player.session[player.idx].amount;
  render();
}

function toggleTimer() {
  if (player.running) {
    stopTimer();
    document.getElementById('timer-btn').textContent = 'Fortsätt';
    return;
  }
  player.running = true;
  document.getElementById('timer-btn').textContent = 'Pausa';
  timerId = setInterval(() => {
    player.remaining -= 1;
    const el = document.getElementById('timer-text');
    if (el) el.textContent = formatSeconds(Math.max(0, player.remaining));
    if (player.remaining <= 0) {
      beep();
      completeSet();
    }
  }, 1000);
}

function beep() {
  try {
    if (navigator.vibrate) navigator.vibrate(200);
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.frequency.value = 880;
    g.gain.value = 0.08;
    o.connect(g).connect(ctx.destination);
    o.start();
    o.stop(ctx.currentTime + 0.25);
    o.onended = () => ctx.close();
  } catch {
    /* ljud är inte kritiskt */
  }
}

// ---------- Feedback / loggning ----------

function viewFeedback(state, { template, minutes, title }) {
  const label = TEMPLATE_LABELS[template]?.name || title || 'Pass';
  return html`
    <div class="stack">
      <div class="card center"><div class="hero-emoji">🎉</div><h1>Bra jobbat!</h1><p class="muted">${label} · ca ${minutes} min</p></div>
      <form class="card" id="feedback" data-template="${template}" data-minutes="${minutes}">
        <div class="field"><span>Hur kändes det?</span>
          <div class="feel">${FEELINGS.map((f, i) => html`<label><input type="radio" name="feeling" value="${f.id}" ${i === 0 ? 'checked' : ''} /><span class="em">${f.em}</span>${f.label}</label>`)}</div>
        </div>
        <div class="field"><span>Märkte du något av detta?</span>
          ${SYMPTOMS.map((s) => html`<label class="check"><input type="checkbox" name="symptoms" value="${s.id}" /> <span>${s.label}</span></label>`)}
        </div>
        <label class="field"><span>Anteckning <span class="muted small">(frivilligt)</span></span><textarea name="note" maxlength="300"></textarea></label>
        <button class="btn primary block" type="submit">Spara passet</button>
      </form>
    </div>`;
}

function viewWalk(state) {
  const { phase } = currentPhase(state.profile);
  return html`
    <div class="stack">
      <div class="card"><h1>Promenad</h1><p>Rekommenderat just nu: <strong>${phase.walkMinutes[0]}–${phase.walkMinutes[1]} minuter</strong>. Vagnen räknas. Flera korta är lika bra som en lång.</p>
      <p class="muted small">Om du känner tyngd i underlivet eller blöder mer efteråt – gör nästa promenad kortare.</p></div>
      <form class="card" id="feedback" data-template="promenad">
        <label class="field"><span>Hur länge gick du?</span><input type="number" name="minutes" min="1" max="300" value="${phase.walkMinutes[0]}" inputmode="numeric" required /></label>
        <div class="field"><span>Hur kändes det?</span>
          <div class="feel">${FEELINGS.map((f, i) => html`<label><input type="radio" name="feeling" value="${f.id}" ${i === 0 ? 'checked' : ''} /><span class="em">${f.em}</span>${f.label}</label>`)}</div>
        </div>
        <div class="field"><span>Märkte du något av detta?</span>
          ${SYMPTOMS.map((s) => html`<label class="check"><input type="checkbox" name="symptoms" value="${s.id}" /> <span>${s.label}</span></label>`)}
        </div>
        <button class="btn primary block" type="submit">Logga promenaden</button>
      </form>
    </div>`;
}

function runWeekIndex(state) {
  const count = state.sessions.filter((s) => s.template === 'lopning').length;
  return Math.min(RUN_PROGRESSION.length - 1, Math.floor(count / 2));
}

function viewRun(state) {
  if (!runReady(state.runChecklist)) return go('learn/lopning');
  const step = RUN_PROGRESSION[runWeekIndex(state)];
  const total = step.rounds * (step.walk + step.run) + 10;
  return html`
    <div class="stack">
      <div class="card">
        <span class="chip accent">Vecka ${step.week} av ${RUN_PROGRESSION.length}</span>
        <h1 style="margin-top:8px">Gå/jogga-intervaller</h1>
        <p>Värm upp med <strong>5 min rask promenad</strong>. Sedan:</p>
        <p class="big-count">${step.walk > 0 ? `${step.run} min jogg · ${step.walk} min gång` : `${step.run} min jogg`}</p>
        <p class="center muted">${step.rounds > 1 ? `× ${step.rounds} omgångar` : 'sammanhängande'} · ca ${total} min totalt</p>
        <p class="small">Avsluta med 5 min lugn gång. Mjukt underlag, lugnt tempo – du ska kunna prata. Två pass per vecka, sedan nästa steg. Symtom under eller efter passet betyder: gör om samma vecka.</p>
      </div>
      <form class="card" id="feedback" data-template="lopning" data-minutes="${total}">
        <div class="field"><span>Hur kändes det?</span>
          <div class="feel">${FEELINGS.map((f, i) => html`<label><input type="radio" name="feeling" value="${f.id}" ${i === 0 ? 'checked' : ''} /><span class="em">${f.em}</span>${f.label}</label>`)}</div>
        </div>
        <div class="field"><span>Märkte du något av detta?</span>
          ${SYMPTOMS.map((s) => html`<label class="check"><input type="checkbox" name="symptoms" value="${s.id}" /> <span>${s.label}</span></label>`)}
        </div>
        <button class="btn primary block" type="submit">Logga passet</button>
      </form>
    </div>`;
}

function saveFeedback(form) {
  const fd = new FormData(form);
  const template = form.dataset.template;
  const minutes = Number(fd.get('minutes') || form.dataset.minutes || 0);
  const symptoms = fd.getAll('symptoms');
  const { phase } = currentPhase(load().profile);
  update((s) => {
    s.sessions.push({
      id: uid(),
      date: toISODate(),
      phaseId: phase.id,
      template,
      minutes,
      feeling: fd.get('feeling') || 'okej',
      symptoms,
      note: String(fd.get('note') || '').trim(),
    });
  });
  if (player) player.finished = 'saved';
  const flagged = symptoms.filter((id) => id !== 'tired');
  if (flagged.length) {
    const advice = SYMPTOMS.find((s) => s.id === flagged[0]).advice;
    toast(advice, 6000);
  } else {
    toast('Sparat. Bra jobbat! 💛');
  }
  go('home');
}

// ---------- Program ----------

function viewProgram(state) {
  const p = state.profile;
  const info = currentPhase(p);
  const cats = Object.keys(CATEGORY_LABELS);
  return html`
    <div class="stack">
      <h1>Programmet</h1>
      <p class="muted">Fyra faser som följer kroppens läkning. Tiderna är riktmärken – din kropp bestämmer takten.</p>
      ${PHASES.map((ph) => {
        const status = ph.index < info.phase.index ? 'klar' : ph.index === info.phase.index ? 'aktuell' : 'kommande';
        const start = phaseStartWeek(ph, p.delivery);
        const std = buildSession(ph.id, 'standard', ph.index === info.phase.index ? info.weekInPhase : 0);
        return html`
          <section class="card" style="border-left:6px solid ${ph.color}">
            <div class="card-head"><h2>${ph.short}: ${ph.name}</h2><span class="chip ${status === 'aktuell' ? 'accent' : status === 'klar' ? 'ok' : ''}">${status === 'aktuell' ? 'Här är du' : status === 'klar' ? 'Klar' : `Från vecka ${start}`}</span></div>
            <p class="muted small">Från vecka ${start}${ph.requiresClearance ? ' · kräver klartecken från efterkontroll' : ''}</p>
            <p>${ph.summary}</p>
            <details><summary>Mål i fasen</summary><div class="body"><ul>${ph.goals.map((g) => html`<li>${g}</li>`)}</ul></div></details>
            <details><summary>Vänta med</summary><div class="body"><ul>${ph.avoid.map((g) => html`<li>${g}</li>`)}</ul></div></details>
            <details><summary>Standardpasset (${estimateMinutes(std)} min)</summary><div class="body"><ul class="ex-list">${std.map((e) => html`<li><span>${e.name}</span><span class="dose">${describeDose(e)}</span></li>`)}</ul></div></details>
          </section>`;
      })}
      <section class="card">
        <h2>Övningsbibliotek</h2>
        ${cats.map((c) => {
          const list = Object.entries(EXERCISES).filter(([, e]) => e.category === c);
          if (!list.length) return '';
          return html`<h3 style="margin-top:12px">${CATEGORY_LABELS[c]}</h3>${list.map(([id, e]) => html`<details><summary>${e.name}</summary><div class="body"><p>${e.cue}</p><p class="muted small">${e.why}</p>${e.caution ? html`<p class="small" style="color:var(--warn)">⚠️ ${e.caution}</p>` : ''}</div></details>`)}`;
        })}
      </section>
    </div>`;
}

// ---------- Logg ----------

function viewLog(state) {
  const week = lastWeek(state.sessions, state.kegelDays);
  const today = toISODate();
  const sessions = [...state.sessions].reverse();
  const evalu = evaluateSymptoms(state.sessions.slice(-3));
  const all = totalMinutes(state.sessions);
  return html`
    <div class="stack">
      <h1>Din logg</h1>
      <section class="stats">
        <div class="stat"><div class="n">${streak(state.sessions, state.kegelDays)}</div><div class="l">dagar i rad</div></div>
        <div class="stat"><div class="n">${state.sessions.length}</div><div class="l">pass totalt</div></div>
        <div class="stat"><div class="n">${all}</div><div class="l">minuter totalt</div></div>
      </section>
      <section class="card">
        <h2>Senaste 7 dagarna</h2>
        <div class="week">${week.map((d) => html`<div class="day ${d.sessions.length || d.kegel ? 'done' : ''} ${d.date === today ? 'today' : ''}"><div class="b">${d.sessions.length ? '✓' : d.kegel ? '·' : ''}</div>${d.weekday}</div>`)}</div>
        <p class="muted small" style="margin-top:8px">✓ = pass, · = bara knip</p>
      </section>
      ${evalu.level !== 'ok' ? html`<div class="banner ${evalu.level === 'contact' ? 'danger' : 'warn'}"><span class="icon">💡</span><div>${evalu.message}</div></div>` : ''}
      <section class="card">
        <h2>Alla pass</h2>
        ${sessions.length
          ? sessions.map((s) => {
              const f = FEELINGS.find((x) => x.id === s.feeling) || FEELINGS[1];
              const ph = PHASES.find((x) => x.id === s.phaseId);
              const sym = (s.symptoms || []).map((id) => SYMPTOMS.find((x) => x.id === id)?.label).filter(Boolean);
              return html`<div class="log-item"><span class="em" aria-label="${f.label}">${f.em}</span><div><strong>${TEMPLATE_LABELS[s.template]?.name || s.template}</strong> · ${s.minutes} min<div class="meta">${formatDate(s.date)} · ${ph ? ph.short : ''}${sym.length ? ` · ${sym.join(', ')}` : ''}</div>${s.note ? html`<div class="small">${s.note}</div>` : ''}</div><button class="btn ghost sm del" data-action="delete-session" data-id="${s.id}" aria-label="Ta bort">✕</button></div>`;
            })
          : html`<p class="muted">Inga pass ännu. Det första kan vara två minuters andning.</p>`}
      </section>
    </div>`;
}

// ---------- Lär dig ----------

function viewLearn(state, route) {
  const open = route.a;
  return html`
    <div class="stack">
      <h1>Lär dig</h1>
      <p class="muted small">Allmän information som stöd. Den ersätter inte råd från barnmorska, läkare eller fysioterapeut.</p>
      ${LEARN.map(
        (c) => html`<details class="card" id="learn-${c.id}" ${open === c.id ? 'open' : ''} style="border-top:1px solid var(--line)"><summary><span>${c.emoji} ${c.title}</span></summary><div class="body">
          ${c.body.map((p) => html`<p>${p}</p>`)}
          ${c.list ? html`<ul>${c.list.map((l) => html`<li>${l}</li>`)}</ul>` : ''}
          ${c.id === 'lopning' ? runChecklistHtml(state) : ''}
          ${c.id === 'varning' ? html`<p class="small"><a href="https://www.1177.se" target="_blank" rel="noopener">1177.se</a> – ring 1177 för sjukvårdsrådgivning, 112 vid akut fara.</p>` : ''}
        </div></details>`
      )}
    </div>`;
}

function runChecklistHtml(state) {
  const checked = state.runChecklist || {};
  const ready = runReady(checked);
  return html`
    <h3 style="margin-top:12px">Checklista innan du joggar</h3>
    ${RUN_CHECKLIST.map((c) => html`<label class="check"><input type="checkbox" data-action="run-check" data-id="${c.id}" ${checked[c.id] ? 'checked' : ''} /> <span>${c.label}</span></label>`)}
    ${ready ? html`<div class="banner ok" style="margin-top:8px"><span class="icon">✅</span><div>Checklistan är klar. Gå/jogga-intervaller finns nu på startsidan när du är i fas 4.</div></div>` : html`<p class="muted small">Alla punkter behöver vara klara. Det är ingen tävling – kroppen kommer dit.</p>`}
    <h3 style="margin-top:14px">Progression (två pass per vecka)</h3>
    <table><thead><tr><th>Vecka</th><th>Jogg</th><th>Gång</th><th>Omg.</th></tr></thead><tbody>
      ${RUN_PROGRESSION.map((r) => html`<tr><td>${r.week}</td><td>${r.run} min</td><td>${r.walk ? `${r.walk} min` : '–'}</td><td>${r.rounds}</td></tr>`)}
    </tbody></table>`;
}

// ---------- Mer / inställningar ----------

function viewMore(state) {
  const p = state.profile;
  const maxPhase = Number.isInteger(p.maxPhase) ? p.maxPhase : 3;
  return html`
    <div class="stack">
      <h1>Mer</h1>
      <form class="card" id="profile">
        <h2>Din profil</h2>
        <label class="field"><span>Namn</span><input type="text" name="name" value="${p.name || ''}" maxlength="30" /></label>
        <label class="field"><span>Förlossningsdatum</span><input type="date" name="birthDate" value="${p.birthDate}" max="${toISODate()}" required /></label>
        <div class="field"><span>Förlossningssätt</span>
          <div class="choice">
            <label><input type="radio" name="delivery" value="vaginal" ${p.delivery !== 'kejsarsnitt' ? 'checked' : ''} /> <span>Vaginal förlossning</span></label>
            <label><input type="radio" name="delivery" value="kejsarsnitt" ${p.delivery === 'kejsarsnitt' ? 'checked' : ''} /> <span>Kejsarsnitt</span></label>
          </div>
        </div>
        <label class="check"><input type="checkbox" name="cleared" ${p.cleared ? 'checked' : ''} /> <span>Jag har fått klartecken att träna vid efterkontrollen</span></label>
        <label class="field" style="margin-top:10px"><span>Tempo</span>
          <select name="maxPhase">
            <option value="3" ${maxPhase >= 3 ? 'selected' : ''}>Följ programmet</option>
            <option value="0" ${maxPhase === 0 ? 'selected' : ''}>Stanna i fas 1 (vila & andning)</option>
            <option value="1" ${maxPhase === 1 ? 'selected' : ''}>Stanna i fas 2 (återhämtning)</option>
            <option value="2" ${maxPhase === 2 ? 'selected' : ''}>Stanna i fas 3 (grund)</option>
          </select>
          <span class="hint">Vill du ta det lugnare ett tag? Välj en fas att stanna i.</span></label>
        <button class="btn primary block" type="submit">Spara</button>
      </form>

      <section class="card">
        <h2>Varningssignaler</h2>
        <p class="small">Pausa träningen och kontakta vården om du får:</p>
        <ul class="small">${RED_FLAGS.map((r) => html`<li>${r.label}</li>`)}</ul>
      </section>

      <section class="card">
        <h2>Din data</h2>
        <p class="muted small">Allt sparas bara på den här enheten. Exportera för att flytta till en annan.</p>
        <div class="btn-row">
          <button class="btn" data-action="export">Exportera</button>
          <label class="btn">Importera<input type="file" accept="application/json" id="import-file" hidden /></label>
        </div>
        <button class="btn danger block" style="margin-top:10px" data-action="reset">Radera allt och börja om</button>
      </section>

      <section class="card soft small">
        <h3>Om appen</h3>
        <p>Igång igen är ett stöd för att komma igång med rörelse efter förlossningen. Innehållet bygger på allmänna riktlinjer för träning efter graviditet och ersätter inte individuella råd från barnmorska, läkare eller fysioterapeut. Lyssna på din kropp.</p>
        <p class="muted">Version 0.1 · ingen data lämnar din enhet.</p>
      </section>
    </div>`;
}

// ---------- Händelser ----------

function afterRender(route, state) {
  const view = document.getElementById('view');
  const onboarding = view.querySelector('#onboarding');
  if (onboarding) {
    onboarding.addEventListener('submit', (e) => {
      e.preventDefault();
      const fd = new FormData(onboarding);
      update((s) => {
        s.profile = {
          name: String(fd.get('name') || '').trim(),
          birthDate: fd.get('birthDate'),
          delivery: fd.get('delivery'),
          cleared: fd.get('cleared') === 'on',
          maxPhase: 3,
          createdAt: toISODate(),
        };
      });
      toast('Välkommen! Börja med det som känns bra i dag.');
      go('home');
    });
  }
  const feedback = view.querySelector('#feedback');
  if (feedback) {
    feedback.addEventListener('submit', (e) => {
      e.preventDefault();
      saveFeedback(feedback);
    });
  }
  const profile = view.querySelector('#profile');
  if (profile) {
    profile.addEventListener('submit', (e) => {
      e.preventDefault();
      const fd = new FormData(profile);
      update((s) => {
        s.profile = {
          ...s.profile,
          name: String(fd.get('name') || '').trim(),
          birthDate: fd.get('birthDate'),
          delivery: fd.get('delivery'),
          cleared: fd.get('cleared') === 'on',
          maxPhase: Number(fd.get('maxPhase')),
        };
      });
      toast('Sparat.');
      render();
    });
  }
  const importFile = view.querySelector('#import-file');
  if (importFile) {
    importFile.addEventListener('change', async () => {
      const file = importFile.files[0];
      if (!file) return;
      try {
        importJSON(await file.text());
        toast('Importerat.');
        render();
      } catch {
        toast('Kunde inte läsa filen.');
      }
    });
  }
  view.querySelectorAll('[data-action="run-check"]').forEach((cb) => {
    cb.addEventListener('change', () => {
      update((s) => {
        s.runChecklist = { ...(s.runChecklist || {}), [cb.dataset.id]: cb.checked };
      });
      render();
      const el = document.getElementById('learn-lopning');
      if (el) el.scrollIntoView();
    });
  });
  if (route.name === 'session' && state.profile) {
    // Timer-knappen ska inte trigga "sluta" av misstag: inget extra att göra.
  }
}

document.getElementById('view').addEventListener('click', (e) => {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;
  if (action === 'run-check') return; // hanteras av change
  const today = toISODate();
  switch (action) {
    case 'start':
      initPlayer(load(), btn.dataset.template);
      go(`session/${btn.dataset.template}`);
      break;
    case 'abort':
      stopTimer();
      player = null;
      break;
    case 'toggle-timer':
      toggleTimer();
      break;
    case 'complete-set':
      completeSet();
      break;
    case 'next':
      nextExercise();
      break;
    case 'prev':
      prevExercise();
      break;
    case 'kegel':
      update((s) => {
        s.kegelDays[today] = (s.kegelDays[today] || 0) + 1;
      });
      toast((load().kegelDays[today] >= KEGEL_GOAL ? 'Dagens knip klara! 🪷' : 'Noterat. Kom ihåg att slappna av mellan.'));
      render();
      break;
    case 'kegel-undo':
      update((s) => {
        s.kegelDays[today] = Math.max(0, (s.kegelDays[today] || 0) - 1);
      });
      render();
      break;
    case 'set-cleared':
      update((s) => {
        s.profile.cleared = true;
      });
      toast('Fas 3 är upplåst. Öka lugnt.');
      render();
      break;
    case 'delete-session':
      update((s) => {
        s.sessions = s.sessions.filter((x) => x.id !== btn.dataset.id);
      });
      render();
      break;
    case 'export': {
      const blob = new Blob([exportJSON()], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `igang-igen-${today}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
      break;
    }
    case 'reset':
      if (confirm('Radera all data och börja om? Det går inte att ångra.')) {
        reset();
        player = null;
        go('onboarding');
        render();
      }
      break;
    default:
      break;
  }
});

window.addEventListener('hashchange', render);
render();

if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  });
}
