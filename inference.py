import streamlit
import requests
import streamlit.components.v1 as components
import json

@streamlit.cache
def load_result(API_URL:str, question:str, context:str, return_raw = 0):
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    response = requests.post(
                            url=API_URL,
                            headers=headers, 
                            json={  "context" : context,
                                    "question" : question,
                                    "return_raw" : return_raw,
                                    })
    print(response.text)
    resp = response.json()[0]
    if return_raw:
        return json.loads(resp)
    return resp
def inference(API_URL :str, default_question :str = None, default_context :str = None):
    """
    default_question : if None, we open input stage. else dropdown selecter
    """
    if default_context is not None: inference.context = default_context

    inference.context = streamlit.text_area(label = "context", value=inference.context, height = 300,
        help="A string that contains the answer to the question.", )
    if default_question is None:
        inference.question = streamlit.text_area(label = "question", value=inference.question,
            help="A string that contains the answer to the question.", )
    else: inference.question = default_question
    ret = load_result(API_URL, inference.question, inference.context)
    streamlit.write(ret)
inference.question = ""
inference.context = ""

def raw_based_inference(API_URL :str, default_question :str = None, default_context :str = None):
    """
    default_question : if None, we open input stage. else dropdown selecter
    """
    if default_context is not None: raw_based_inference.context = default_context

    raw_based_inference.context = streamlit.text_area(label = "context", value=raw_based_inference.context, height = 300,
        help="A string that contains the answer to the question.", )
    if default_question is None:
        raw_based_inference.question = streamlit.text_area(label = "question", value=raw_based_inference.question,
            help="A string that contains the answer to the question.", )
    else: raw_based_inference.question = default_question

    infos = load_result(API_URL, raw_based_inference.question, raw_based_inference.context, return_raw=1)
    infos = dict(infos)

    total_tokens = len(infos["start_prob"])
    # build pair 
    probs = [(I*J, (s,e)) for s, I in enumerate(infos["start_prob"]) for e, J in enumerate(infos["end_prob"])]
    probs.sort()
    probs = probs[::-1]

    prob_min = streamlit.sidebar.slider("minimum probability", min_value=0., max_value=probs[0][0], value=0.)
    ANS_length = streamlit.sidebar.slider("answer token count limit", min_value=8, max_value=total_tokens, value=50, step = 8)
    ANS_count = streamlit.sidebar.slider("how many answers you want", min_value=1, max_value=16, value=4, step = 1)
    
    """
    infos["start_prob"]
    infos["end_prob"]

    infos["token_start_index"]
    infos["token_end_index"]
    """
    # conditional 
    probs = [(P, SE) for P, SE in probs if prob_min < P and SE[1] - SE[0] + 1 <= ANS_length]
    probs = probs[:ANS_count]

    if len(probs) == 0: 
        streamlit.warning("no candidates are exist with your option")
        return 

    options = []
    for P, (ST, ET) in probs:
        original_string = raw_based_inference.context[infos["token_start_index"][ST]:infos["token_end_index"][ET]]
        
        prob_info = ("%." + "%df"%(6 - len(f"{P*100:.1f}")))%(P*100)
        prob_info = f"P={prob_info}%"
        options.append(f'{prob_info} | {original_string}')
        
    option_idx = streamlit.selectbox("which cadidate you want to see?", list(range(len(options))), format_func=(lambda idx: options[idx]))
    option = options[option_idx]
    
    P, (ST, ET) = probs[option_idx]
    pre = raw_based_inference.context[:infos["token_start_index"][ST]]
    inter = raw_based_inference.context[infos["token_start_index"][ST]:infos["token_end_index"][ET]]
    post = raw_based_inference.context[infos["token_end_index"][ET]:]
    components.html(
        f'<span style="color:grey">{pre}<span style="color:black">{inter}</span>{post}</span>',
        height=1000
    )
raw_based_inference.question = ""
raw_based_inference.context = ""