// Publication density only. Keep every measured point, assignment and full text.
(() => {
  const make = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const scrollBox = node => {
    if (node.parentElement?.classList.contains('paper-table-scroll')) return;
    const box = make('div', 'paper-table-scroll');
    node.before(box); box.append(node);
    box.tabIndex = 0;
  };
  const foldExtra = (nodes, limit, label) => {
    const extra = nodes.slice(limit);
    if (!extra.length) return null;
    const fold = make('details', 'paper-more-content');
    fold.append(make('summary', '', `${label} (${extra.length})`));
    extra[0].before(fold);
    const body = make('div', 'paper-more-body');
    body.append(...extra); fold.append(body);
    return fold;
  };
  const foldProse = (node, words, lines) => {
    if (!node || node.textContent.trim().split(/\s+/).length <= words || node.closest('details')) return;
    const fold = make('details', 'paper-prose-fold');
    const summary = make('summary');
    const preview = node.cloneNode(true);
    preview.classList.add('paper-prose-preview');
    preview.style.setProperty('--paper-preview-lines', lines);
    summary.append(preview, make('span', 'paper-read-more', 'Read full text'));
    node.before(fold); fold.append(summary, node);
  };

  const style = make('style');
  style.textContent = `
    .paper-more-content{margin:16px 0}
    .paper-more-content>summary,.paper-prose-fold>summary{cursor:pointer;color:var(--ink3,#777)}
    .paper-more-body{padding-top:12px}
    .paper-table-scroll{max-height:560px;overflow:auto;overscroll-behavior:contain;max-width:100%}
    .paper-table-scroll th{position:sticky;top:0;background:var(--bg,#fff);z-index:1}
    .paper-more-groups .paper-table-scroll{max-height:420px}
    .paper-more-groups table{width:100%}
    #anatomy .legend{max-height:210px;overflow:auto}
    #anatomy .legend>[hidden]{display:none!important}
    .paper-label-preview{display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;overflow:hidden}
    .paper-prose-fold{margin:0 0 14px}
    .paper-prose-preview{display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:var(--paper-preview-lines,8);overflow:hidden;color:var(--ink2,#4a4a4f)}
    .paper-read-more{display:block;font-size:12px;margin-top:5px}
    .paper-prose-fold[open]>summary .paper-prose-preview{display:none}
    .paper-prose-fold[open]>summary .paper-read-more{font-size:0}
    .paper-prose-fold[open]>summary .paper-read-more::after{content:'Hide full text';font-size:12px}
    #family .ncol .chip{max-width:100%}
    #family .ncol .chip .label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  `;
  document.head.append(style);

  const anatomy = document.getElementById('anatomy');
  const table = anatomy?.querySelector('.qtab');
  if (table && table.rows.length > 9) {
    const rows = [...table.rows].slice(1);
    const fold = make('details', 'paper-more-content paper-more-groups');
    fold.append(make('summary', '', `More semantic groups (${rows.length - 8})`));
    const rest = table.cloneNode(false);
    rest.append(table.rows[0].cloneNode(true), ...rows.slice(8));
    fold.append(rest); table.after(fold); scrollBox(rest);
    const legend = [...(anatomy.querySelector('.legend')?.children || [])];
    legend.slice(8).forEach(node => { node.hidden = true; });
    fold.addEventListener('toggle', () => {
      legend.slice(8).forEach(node => { node.hidden = !fold.open; });
    });
  }
  anatomy?.querySelectorAll('.qtab tr td:first-child').forEach(cell => {
    cell.title = cell.textContent;
    const contents = make('span', 'paper-label-preview');
    contents.append(...cell.childNodes); cell.append(contents);
  });

  foldExtra([...document.querySelectorAll('#custom > .figs > .cfig')], 3, 'Additional figures');
  foldExtra([...document.querySelectorAll('#clusters .card')], 8, 'Additional clusters');
  foldExtra([...document.querySelectorAll('#criticism .crow')], 8, 'Additional criticism rounds');

  document.querySelectorAll('#ablation table,#tests table').forEach(scrollBox);
  document.querySelectorAll('#circuit table tr td:nth-child(2)').forEach(cell => {
    cell.title = cell.textContent;
    const contents = make('span', 'paper-label-preview');
    contents.append(...cell.childNodes); cell.append(contents);
  });
  document.querySelectorAll('main>section>.text>p').forEach(node => foldProse(node, 130, 7));
  document.querySelectorAll('#clusters .story').forEach(node => foldProse(node, 80, 4));
  document.querySelectorAll('#agents-read p:not(.prf-meta)').forEach(node => foldProse(node, 130, 7));
  document.querySelectorAll('#agents-read h3').forEach(heading => {
    const entries=[];
    for (let node=heading.nextElementSibling; node?.classList.contains('prf-item'); node=node.nextElementSibling) entries.push(node);
    foldExtra(entries,3,'Additional observations');
  });
  // Numeric activation badges encode sign; token shading continues to encode magnitude.
  const activationBadge = node => {
    const value=Number(node.textContent.trim().replaceAll(',',''));
    if (!Number.isFinite(value)) return;
    node.style.background=value<0?NEG:POS;
    node.style.color='#fff';
    node.title=`Neuron activation ${value}. Badge colour indicates sign.`;
  };
  document.querySelectorAll('#search .ex>.tile,#search .best .tile,#clusters .ex>.tile').forEach(activationBadge);
  document.querySelectorAll('#clusters .tiles .k').forEach(label => {
    if (/^typical activation$/i.test(label.textContent.trim())) {
      const badge=label.parentElement.querySelector('.tile');if(badge)activationBadge(badge);
    }
  });
  document.addEventListener('toggle',event=>{
    if(event.target.open && event.target.closest('#search,#clusters')) {
      event.target.querySelectorAll('.ex>.tile').forEach(activationBadge);
    }
  },true);
})();
