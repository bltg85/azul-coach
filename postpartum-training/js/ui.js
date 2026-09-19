// Små hjälpfunktioner för att rendera HTML säkert.

export function esc(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Sträng som redan är säker HTML (resultat av html`` eller raw()). */
class Safe extends String {}

/** Taggad mall: interpolerade värden escapas, om de inte redan är säkra (html`` eller raw()). */
export function html(strings, ...values) {
  return new Safe(
    strings.reduce((out, str, i) => out + render(values[i - 1]) + str)
  );
}

function render(v) {
  if (v == null || v === false) return '';
  if (v instanceof Safe) return v.toString();
  if (Array.isArray(v)) return v.map(render).join('');
  return esc(v);
}

export function raw(str) {
  return new Safe(String(str ?? ''));
}

let toastTimer = null;
export function toast(message, ms = 2200) {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), ms);
}

export function formatDate(iso) {
  const [y, m, d] = iso.split('-').map(Number);
  return new Date(y, m - 1, d).toLocaleDateString('sv-SE', { weekday: 'short', day: 'numeric', month: 'short' });
}

export function pad(n) {
  return String(n).padStart(2, '0');
}

export function formatSeconds(s) {
  return `${pad(Math.floor(s / 60))}:${pad(s % 60)}`;
}

export function uid() {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}
