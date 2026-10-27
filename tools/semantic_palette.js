// One semantic colour mapping for the spectrum, clusters and gate/value scatter.
const semanticPalette = () => {
  const groups = new Map();
  for (const group of [...(B.spectrum?.groups || []), ...(B.anatomy?.semantic_groups || [])]) {
    if (!groups.has(group.id)) groups.set(group.id, group);
  }
  return polePalette([...groups.values()]);
};
