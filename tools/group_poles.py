"""Summarize measured response signs for saved semantic groups."""
import copy
from pathlib import Path
from common import write
from measured_response import annotate_spectrum, describe_group


def annotate(spectrum, folder, llm=None):
    result = annotate_spectrum(copy.deepcopy(spectrum))
    folder = Path(folder)
    write(folder/'spectrum.json', result)
    write(folder/'result.json', {'status':'complete', 'llm_requests':0, 'new_model_forwards':0})
    return result


def cluster_poles(clusters, spectrum):
    result = []
    for cluster in clusters:
        members = cluster.get('member_groups', [cluster['id']])
        result.append(describe_group(copy.deepcopy(cluster),
            [p['activation'] for p in spectrum.get('points', []) if p['group'] in members]))
    return result
