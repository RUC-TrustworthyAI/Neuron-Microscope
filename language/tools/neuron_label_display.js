const neuronLabelDisplay = (() => {
  const data = JSON.parse(document.getElementById('neuron-labels').textContent);
  const idOf = u => u && `L${u.layer}_N${u.neuron}`;
  const info = u => data.labels[idOf(u)];
  const name = (u, fallback) => {
    const label = info(u);
    if (!label) return fallback;
    return label.status === 'labelled' && label.label ? label.label : 'Insufficient evidence';
  };
  const graphUnit = n => {
    const kind = n.feature_type ?? n.type;
    if (kind && !['mlp_neuron', 'mlp_neuron_target', 'neuron'].includes(kind)) return null;
    const unit = {layer: n.layer, neuron: n.feature ?? n.neuron};
    return /^L\d+_N\d+$/.test(idOf(unit)) ? unit : null;
  };

  function show(unit) {
    const label = info(unit);
    const id = idOf(unit);
    let dialog = document.getElementById('neuron-label-evidence');
    if (!dialog) {
      dialog = $('dialog', {id: 'neuron-label-evidence', 'aria-labelledby': 'neuron-label-title'});
      document.body.append(dialog);
    }
    const close = $('button', {class: 'more neuron-label-close', type: 'button'}, 'Close');
    close.onclick = () => dialog.close();
    dialog.replaceChildren($('div', null, close,
      $('div', {class: 'neuron-label-meta'}, id),
      $('h2', {id: 'neuron-label-title'}, name(unit)),
      label.description ? P(label.description) : null,
      label.positive ? P($('b', null, 'Positive response. '), label.positive) : null,
      label.negative ? P($('b', null, 'Negative response. '), label.negative) : null));
    const samples = label.evidence_ids.map(key => data.evidence[key]).filter(row => row?.neuron_id === id);
    if (samples.length) {
      dialog.append($('h3', {class: 'sub'}, 'Measured examples'), $('ul', null, samples.map(row =>
        $('li', null, $('div', {class: 'neuron-label-meta'},
          [row.document != null ? 'Document ' + row.document : null, `Position ${row.position}`,
            `Token ${JSON.stringify(row.token)}`, row.token_id != null ? `Token ID ${row.token_id}` : null,
            `Product ${row.activation}`, row.id]
            .filter(value => value != null).join(' · ')),
        $('div', {class: 'neuron-label-context'}, row.prefix, $('mark', null, row.token), row.suffix)))));
    }
    if (!dialog.open) dialog.showModal();
    close.focus();
  }

  function keyboard(element, activate) {
    element.setAttribute('role', 'button');
    element.setAttribute('tabindex', '0');
    element.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      activate(event);
    });
  }

  function evidenceControl(element, unit) {
    element.classList.add('neuron-label-control');
    element.setAttribute('aria-label', `${name(unit)} (${idOf(unit)}) measured evidence`);
    element.setAttribute('aria-haspopup', 'dialog');
    element.addEventListener('click', () => show(unit));
    if (element.tagName !== 'BUTTON') keyboard(element, () => show(unit));
    return element;
  }

  function chip(unit, element) {
    if (!info(unit)) return element;
    const text = name(unit);
    const oldLabel = element.querySelector('.lbl');
    const label = $('span', {class: 'lbl'}, text);
    if (oldLabel) oldLabel.replaceWith(label);
    else element.querySelector('.id').after(label);
    element.title = text + (unit.sign < 0 ? ' (feeds through its negative pole)' : '');
    return evidenceControl(element, unit);
  }

  function heatmap(svg, units) {
    const labels = svg.querySelectorAll('text[font-size="12"]');
    units.forEach((unit, i) => {
      if (!info(unit) || !labels[i]) return;
      const text = `L${unit.layer}/N${unit.neuron} · ${name(unit)}`;
      labels[i].textContent = text;
      labels[i].append(svgEl('title', null, text));
      evidenceControl(labels[i], unit);
    });
  }

  function heatmapWidth(units, minimum) {
    return Math.max(minimum, ...units.filter(info).map(u => (15 + name(u).length) * 7));
  }

  function button(unit) {
    return info(unit) ? evidenceControl($('button', {type: 'button'}, name(unit)), unit) : null;
  }

  function cell(unit, element) {
    const control = button(unit);
    if (control) element.replaceChildren(control);
    return element;
  }

  function graphNode(element, n, panel) {
    const unit = graphUnit(n);
    if (!info(unit)) return;
    element.classList.add('neuron-label-control');
    element.setAttribute('aria-label', `${name(unit)} (${idOf(unit)}) at position ${n.ctx_idx}`);
    keyboard(element, event => {
      element.onclick(event);
      panel.querySelector('.neuron-label-control')?.focus();
    });
  }

  const tooltip = value => String(value).replace(/[&<>]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[ch]));
  return {name, chip, heatmap, heatmapWidth, cell, graphUnit, graphNode, graphControl: n => button(graphUnit(n)), tooltip};
})();
