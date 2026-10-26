// Exemplar display context: keep 128 saved tokens before the first highlighted token.
const exemplarDisplay = (example, scale) => {
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
  const start=Math.max(0,Math.min(first<0?peak:first,peak)-128);
  if (!start) return example;
  const visible=tokens.slice(start);
  return {...example,token_activations:visible,peak_index:peak-start,
    text:visible.map(token=>token.token).join(''),cut_left:true};
};
