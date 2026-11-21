(() => {
  const {claim, evidence, peak_census: savedCensus} =
    JSON.parse(document.getElementById('report-synthesis').textContent);
  const headings = Object.hasOwn(claim.findings, 'main_response') ? {
    main_response: '', negative_pole: 'Negative pole',
    what_would_change_my_mind: 'What would change my mind',
  } : {
    pattern: 'Core response', strength: 'Strong and weak responses',
    mechanism: 'Mechanism and output effects', open_questions: 'Open questions',
  };
  const evidencePrefix = '#evidence-';

  function locate(entry, section) {
    if (!section || !entry.locator) return null;
    const {id, row, sample, graph} = entry.locator;
    if (graph) {
      const index = (B.graphs_curated?.picks || []).findIndex(pick => pick.id === graph);
      const tab = section.querySelectorAll('.bands button')[index];
      tab?.click();
      return tab && section.querySelector('.gtop');
    }
    if (entry.section === 'anatomy') {
      return [...section.querySelectorAll('[data-case-id]')].find(node => node.dataset.caseId === String(id));
    }
    if (entry.section === 'ablation') {
      const table = section.querySelector('table[aria-label="Per-example ablations"]');
      if (!table || id == null) return null;
      const ordered = [...(B.ablation?.rows || [])].sort((a, b) => (b.kl ?? 0) - (a.kl ?? 0));
      const index = ordered.findIndex(value => String(value.case_id) === String(id));
      const rows = table.tBodies[0]?.rows;
      if (rows && index >= rows.length) table.closest('figure')?.querySelector('button.more')?.click();
      return rows?.[index];
    }
    if (entry.section === 'tests') {
      if (id === 'blind_max' || id === 'blind_min') {
        const title = id === 'blind_max' ? 'Blind max:' : 'Blind min:';
        return [...section.querySelectorAll('.pair .sub')]
          .find(node => node.textContent.startsWith(title))?.closest('.pair');
      }
      const table = section.querySelector('table[aria-label="Individual route predictions"]');
      const index = (B.tests?.route_prediction?.rows || []).findIndex(value => String(value.id) === String(id));
      return table?.tBodies[0]?.rows[index];
    }
    if (entry.section === 'criticism') {
      const item = section.querySelectorAll('details.crow')[row];
      return sample == null ? item : item?.querySelector('.ev')?.children[sample];
    }
  }

  const sampleCard = $('details', {class: 'detail'});
  function openEvidence(entry) {
    const section = document.getElementById(entry.locator?.graph ? 'graphs' : entry.section);
    let target = locate(entry, section);
    sampleCard.remove();
    if (!target) {
      // A filtered plot or old report may not have a persistent node for the sample.
      const figure = entry.locator?.figure;
      const host = (figure && document.getElementById('custom-' + figure)) ||
        section?.querySelector('.figs') || document.querySelector('#claim .figs');
      sampleCard.replaceChildren($('summary', null, entry.label));
      if (entry.text) {
        if (entry.text.startsWith('…') || entry.text.endsWith('…')) sampleCard.append($('p', {class: 'sub'}, 'Saved text excerpt'));
        sampleCard.append($('p', {style: 'white-space:pre-wrap;overflow-wrap:anywhere'}, entry.text));
      }
      Object.entries(entry.details || {}).forEach(([label, value]) => {
        sampleCard.append(P($('b', null, label.replaceAll('_', ' ')), ' ', String(value)));
      });
      host.prepend(sampleCard);
      target = sampleCard;
    }
    if (entry.section === 'criticism' && entry.locator?.sample != null && target !== sampleCard &&
        !target.querySelector('.scored-evidence')) {
      const {token, position, location: scoringWindow, reduction} = entry.details || {};
      const parts = [
        token != null && `Scored token ${JSON.stringify(token)}`,
        position != null && `position ${position} (0-based)`,
        scoringWindow && scoringWindow.replaceAll('_', ' '),
        reduction === 'absmax' && 'signed largest absolute activation',
      ].filter(Boolean);
      if (parts.length) target.append($('span', {
        class: 'cap scored-evidence', style: 'grid-column:2',
      }, parts.join(' · ')));
    }
    if (target.tagName === 'TR') {
      const disclosure = target.querySelector('details');
      if (disclosure) disclosure.open = true;
    }
    for (let node = target; node; node = node.parentElement) {
      if (node.tagName === 'DETAILS') node.open = true;
    }
    target.tabIndex = -1;
    target.focus({preventScroll: true});
    target.scrollIntoView({block: 'center'});
  }

  function openEvidenceFromHash() {
    if (!location.hash.startsWith(evidencePrefix)) return;
    let id;
    try {
      id = decodeURIComponent(location.hash.slice(evidencePrefix.length));
    } catch {
      return;
    }
    if (Object.hasOwn(evidence, id)) openEvidence(evidence[id]);
  }

  function citations(ids) {
    if (!ids.length) return null;
    const links = ids.flatMap((id, index) => {
      const entry = evidence[id];
      const href = evidencePrefix + encodeURIComponent(id);
      const link = $('a', {href, title: entry.label, 'aria-label': entry.label}, String(index + 1));
      link.addEventListener('click', event => {
        if (location.hash === href && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey) {
          event.preventDefault();
          openEvidence(entry);
        }
      });
      return index ? [', ', link] : [link];
    });
    return [' ', $('sup', null, '[', links, ']')];
  }

  const headline = document.querySelector('#head h1');
  headline.textContent = claim.title;
  document.title = claim.title;
  const lede = document.querySelector('#head .lede');
  if (lede) lede.textContent = claim.lede;
  else headline.after($('p', {class: 'lede'}, claim.lede));
  document.querySelector('#head .metarow .pill')?.parentElement.remove();

  const findingBlocks = Object.entries(headings).flatMap(([key, title]) => {
    const finding = claim.findings[key];
    return finding ? [P(title ? $('b', null, title + '. ') : null, finding.text, citations(finding.evidence_ids))] : [];
  });
  const findings = $('div', {class: 'detail'}, findingBlocks);
  const claimFigures = document.querySelector('#claim .figs');
  const previous = claimFigures?.querySelector('.detail');
  if (previous) previous.replaceWith(findings);
  else if (claimFigures) claimFigures.prepend(findings);
  else {
    sec('claim', 'The claim in detail', null, findings);
    document.getElementById('main').prepend(document.getElementById('claim'));
  }
  if (savedCensus) {
    const figures = document.querySelector('#claim .figs');
    const previousCensus = figures.querySelector('figure:has(.census)');
    const census = peakCensusFigure(combinedPeakCensus(savedCensus.peak_census, savedCensus.peak_census_negative));
    if (census && previousCensus) previousCensus.replaceWith(census);
    else if (census) figures.append(census);
  }

  const guides = {
    custom: 'Compare the conditions within each controlled experiment to see which inputs elicit a response.',
    graphs: 'Switch between inputs to see how attribution flows into the highlighted target.',
    distribution: 'Use the signed activation quantiles to compare typical corpus responses with the extremes.',
    spectrum: 'Compare the contexts and their assigned groups across activation bins.',
    exemplars: 'Compare corpus examples across activation strengths. Start at the outlined token and read its prefix.',
    search: B.search?.source_kind === 'saved_replacement_sweeps'
      ? 'Compare the scored replacements within each saved sweep. Each sweep keeps its original scoring rule.'
      : 'Compare the best measured probe for each hypothesis with the corpus extremes.',
    clusters: 'Compare grouped attribution graphs to see which patterns recur across inputs.',
    anatomy: 'Compare gate, up, and the resulting activation at the highlighted token positions.',
    ablation: 'Read direct logit effects as weight-based scores and ablations as measured changes to the next-token distribution.',
    family: 'Compare the target with related units to see where their activations align on the same inputs.',
    circuit: 'Use the attribution weights to see which units feed the target and which read from it.',
    tests: B.tests?.reused_measurement
      ? 'Read the reused tests against the explanation that originally produced their predictions.'
      : B.tests?.blind_max || B.tests?.blind_min
        ? 'Compare predictions with the saved measurements. Blind tests use the frozen explanation.'
        : 'Compare the tested predictions or interventions with their saved measurements.',
    criticism: 'Read the verdicts and evidence to see how counterexamples affect the explanation.',
    'interpretation-costs': 'Compare measured time and estimated API charges across workflow stages.',
  };
  for (const [id, guide] of Object.entries(guides)) {
    const host = document.getElementById(id)?.querySelector('.text');
    if (!host) continue;
    let why = host.querySelector('.why');
    if (!why) host.append(why = $('div', {class: 'why'}));
    why.replaceChildren(P(guide));
  }

  const figureInputs = Array.isArray(B.custom) ? B.custom : B.custom?.figures || [];
  const conditionName = row => row.tag || row.filler || row.country;
  const score = row => `${conditionName(row)} (${row.activation > 0 ? '+' : ''}${fmt(row.activation)})`;
  for (const figure of figureInputs) {
    if (!figure.id) continue;
    const chart = document.getElementById('custom-' + figure.id)?.querySelector('figure');
    if (!chart) continue;
    const measured = (figure.results || []).filter(row => Number.isFinite(row.activation) && conditionName(row))
      .sort((a, b) => Math.abs(b.activation) - Math.abs(a.activation));
    if (!measured.length) continue;
    const target = measured.find(row => row.expected === 'positive' || row.expected === 'negative');
    const controls = measured.filter(row => row.expected === 'neutral');
    const [control] = controls;
    let note;
    if (target && control) {
      note = `The largest absolute responses among the targeted conditions and planned controls were ${score(target)} and ${score(control)}, respectively.`;
      if (controls.length > 1) note += ` The control closest to zero was ${score(controls.at(-1))}.`;
    } else {
      note = `${score(measured[0])} had the largest absolute response.`;
      if (measured.length > 1) note += ` ${score(measured.at(-1))} was closest to zero.`;
    }
    let caption = chart.querySelector('figcaption');
    if (!caption) chart.append(caption = $('figcaption', {class: 'cap'}));
    caption.prepend($('div', {class: 'figure-readout'}, note));
  }

  const assessment = 'Independent assessment of the frozen explanation';
  for (const item of document.querySelectorAll('#head .pr-k, #agents-read .prf-k')) {
    item.textContent = assessment;
  }
  window.addEventListener('hashchange', openEvidenceFromHash);
  openEvidenceFromHash();
})();
