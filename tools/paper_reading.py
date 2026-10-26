"""One independent reading of the finished paper, using this run's evidence."""
from pathlib import Path
import copy
from common import read, write
from blind_tests import call

GROUP_LABELS = {'one_idea': 'One idea', 'branching': 'One idea, branching', 'unplaced': 'One idea + residual', 'two_related': 'Two related jobs', 'mixture': 'Mixture', 'unresolved': 'Unresolved', 'none': 'No strong firings of its own'}

SYSTEM = """Read a neuron paper and its evidence. Assess the overall organization of this neuron's responses and assess each response pole separately.

Use these seven classification codes:
- one_idea: One explanatory principle accounts for the material responses. Small exceptions do not by themselves establish another job.
- branching: A clear core job has related variants or branches. The evidence supports their relationship to that core.
- unplaced: A clear main job is supported, but some material responses remain outside the account. These are unexplained residuals, rather than established additional jobs.
- two_related: Two distinct, supported jobs are present and have an intelligible relationship, but neither is adequately described as a variant of one core job.
- mixture: Multiple distinct jobs are positively supported and lack an adequate unifying relationship. Heterogeneous or unexplained examples alone do not establish multiple jobs.
- unresolved: The evidence does not support a stable main account or distinguish the competing accounts.
- none: There are no substantial strong responses of the pole being assessed. Use this for the overall result only when neither pole has substantial strong responses.

Overall classification rules:
Assess the positive and negative poles first, then explain their combined organization. Do not take the worse pole's code, vote between the poles, or require them to be clean antonyms.
A well-supported main job with meaningful remaining responses outside its account is unplaced. This includes a clear main account on one pole with substantial unexplained responses on the other.
Use one_idea when a single account covers the material responses of the active poles, and branching when their additional patterns are supported variants of that core.
Use two_related or mixture only when the additional jobs themselves are supported: related distinct jobs give two_related; distinct jobs without a supported relationship give mixture.
An inactive opposite pole does not prevent a coherent active pole from being one_idea. If no main account is supported, use unresolved.
Consider response strength, recurrence and explanatory coverage. Counts in selected extrema or stratified examples are not population frequencies. Explain the chosen codes in one_line and the pole notes.

Give organized and understood scores from 1 to 5 as your assessment, plus what is convincing and what remains unexplained.
Return JSON with group (one of the seven codes), organized, understood, one_line, main_job, remainders (objects with what and evidence), where_it_branches (objects with branch, grows_from and path), concerning (strings), fine (strings), and poles (positive and negative objects with group and notes).
Return classification codes; display names are assigned by the renderer."""

def display_names(review):
    """Map classifier codes to names without changing the model's judgment."""
    review=copy.deepcopy(review)
    if review.get('group') in GROUP_LABELS:
        review['group_name']=GROUP_LABELS[review['group']]
    return review


def evaluate(messages,folder,llm=call):
    folder=Path(folder)
    raw=llm(messages,folder,'paper-reader',max_tokens=393216)
    result=display_names(raw)
    write(folder/'review.json',result)
    return result


def reassess(request_file,output,llm=call):
    """Reuse the exact saved evidence; replace only the review instructions."""
    previous=read(request_file)
    messages=[dict(m,content=SYSTEM) if m['role']=='system' else copy.deepcopy(m)
              for m in previous['messages']]
    return evaluate(messages,output,llm)


def experiment_evidence(measurements):
    # The reader needs results, not duplicated plot rendering payloads.
    def data(value):
        if isinstance(value,dict):
            return {k:data(v) for k,v in value.items()
                    if k not in ['token_activations','input_token_ids','token_ids','readouts','routes_matrix']}
        if isinstance(value,list):return [data(v) for v in value]
        return value
    result=[]
    for measurement in measurements:
        row={k:data(v) for k,v in measurement.items() if k!='measured_sections'}
        sections={}
        for key,value in measurement.get('measured_sections',{}).items():
            if key=='spectrum':
                value={k:v for k,v in value.items() if k!='points'}
            sections[key]=data(value)
        if sections:row['measured_sections']=sections
        result.append(row)
    return result

def run(session, final, llm=call):
    packet=session.prepare()
    contexts=packet['contexts_json']['cases']
    chosen=[]
    for group in ['top_tokens','bottom_tokens']:
        chosen.extend([c for c in contexts if group in c['groups']][:8])
    if not chosen:
        chosen=contexts[:16]
    payload={'claim':final['claim'],'criticism':final.get('criticism',[]),
        'distribution':packet['distribution_json'],
        'contexts':chosen,'experiments':experiment_evidence(final.get('_measurements',[]))}
    messages=[{'role':'system','content':SYSTEM},
        {'role':'user','content':__import__('json').dumps(payload,ensure_ascii=False)}]
    folder=session.root/'paper-reading'
    number=1
    while folder.exists():
        number+=1;folder=session.root/('paper-reading-%03d'%number)
    return evaluate(messages,folder,llm)
