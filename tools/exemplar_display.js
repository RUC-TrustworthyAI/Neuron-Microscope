// Exemplar display context: retain the requested lead-in before the first highlighted token.
function exemplarDisplay(example, scale, before = 128) {
  const tokens = example.token_activations || [];
  if (!tokens.length) return example;
  const peak = Number.isInteger(example.peak_index)
    ? Math.max(0,Math.min(tokens.length-1,example.peak_index))
    : argmaxAbs(tokens.map(token=>token.activation || 0));
  const first = tokens.findIndex(token=>{
    const magnitude=Math.abs(token.activation || 0);
    return magnitude/scale >= 0.10 && magnitude >= 2;
  });
  // Keep the outlined point even if no token in this sample reaches the list's color threshold.
  const start=Math.max(0,Math.min(first<0?peak:first,peak)-before);
  if (!start) return example;
  const visible=tokens.slice(start);
  return {...example,token_activations:visible,peak_index:peak-start,
    text:visible.map(token=>token.token).join(''),cut_left:true};
}

const exemplarPreview = (example, before = 32) => {
  const tokens=example.token_activations || [];
  if (!tokens.length) return example;
  const peak=Number.isInteger(example.peak_index)
    ? Math.max(0,Math.min(tokens.length-1,example.peak_index))
    : argmaxAbs(tokens.map(token=>token.activation || 0));
  const start=Math.max(0,peak-before),end=peak+1,visible=tokens.slice(start,end);
  return {...example,token_activations:visible,peak_index:peak-start,
    text:visible.map(token=>token.token).join(''),
    cut_left:!!example.cut_left || start>0,cut_right:!!example.cut_right || end<tokens.length};
};

const foldedExample = (example,scale,negative,tileScale,before = 32) => {
  const details=$('details',{class:'example-fold'}),body=$('div',{class:'example-full'});
  const label=$('span',{class:'example-toggle'},'Show full context');
  details.append($('summary',null,exRow(exemplarPreview(example,before),scale,negative,tileScale),label),body);
  let built=false;
  details.addEventListener('toggle',()=>{
    label.textContent=details.open?'Hide full context':'Show full context';
    if (details.open && !built) {body.append(exRow(example,scale,negative,tileScale));built=true;}
  });
  return details;
};
