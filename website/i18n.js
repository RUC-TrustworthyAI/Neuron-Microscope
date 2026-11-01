(() => {
'use strict';
const language = new URLSearchParams(location.search).get('lang') === 'zh' ? 'zh' : 'en';
document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
const t = (zh, en) => language === 'zh' ? zh : en;
function updateLanguageLinks() {
  document.querySelectorAll('[data-language]').forEach(a => {
    const url = new URL(location.href); url.searchParams.set('lang', a.dataset.language); a.href = url.href;
    const active = a.dataset.language === language; a.classList.toggle('active', active);
    if (active) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
}
document.querySelectorAll('[data-zh][data-en]').forEach(el => {
  const value = el.dataset[language];
  if (el.hasAttribute('data-html')) el.innerHTML = value; else el.textContent = value;
});
window.NeuronDemoI18n = {language, t, updateLanguageLinks};
updateLanguageLinks();
})();
