"""Display-only context windows for saved Exemplars and Clusters; evidence stays intact."""
from pathlib import Path
import re

HELPER = Path(__file__).with_suffix('.js')
MARKER = '// Exemplar display context:'


def render(source):
    match = re.search(r'// 2\. Exemplars[^\n]*\n', source)
    if match is None:
        return source
    if MARKER not in source:
        source = source[:match.start()] + HELPER.read_text() + '\n' + source[match.start():]
    start = source.find('// 2. Exemplars')
    stop = source.find('// 3. Search', start)
    if stop < 0:
        return source
    block = source[start:stop]
    block = block.replace('e => exRow(e, s, neg, G)',
                          'e => exRow(exemplarDisplay(e, s), s, neg, G)')
    source = source[:start] + block + source[stop:]
    source = source.replace(
        'const exList = (items, negPeak, tileScale) => {',
        'const exList = (items, negPeak, tileScale, displayWindow = false) => {')
    source = source.replace(
        'items.map(e => exRow(e, s, negPeak, tileScale))',
        'items.map(e => exRow(displayWindow ? exemplarDisplay(e, s) : e, s, negPeak, tileScale))')
    source = source.replace(
        "exList(k.examples.slice(0, 3), poleOf(k) === 'negative', exScale)",
        "exList(k.examples.slice(0, 3), poleOf(k) === 'negative', exScale, true)")
    source = source.replace(
        'exList(r.examples.slice(0, 2))',
        'exList(r.examples.slice(0, 2), undefined, undefined, true)')
    return source
