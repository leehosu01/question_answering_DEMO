import streamlit
import inference
samples = [
    {
        "title":"common computer summary",
        "context" : """We are committed to realize the internet of value with blockchain cloud and artificial intelligence.
We are called Common Computer, and our head office is located in Yangjae-dong, Seoul.
We provide Teachable-NLP and Xangle services by automated AI training based on the transformers model. Currently we provide following models for train them : Deberta-base, DistilBert-base, Bart-Base, Bart-Large, KoBART, Electra and GPT2.
Our product `Workspace` is currently in the final stages of commercialization, which will become a competitor to google's colab.""",
        "questions": ["Where do I live?",
         "Where am I?",
         "Who is competitor of google's colab",
        "Who is competitor of Workspace",
        "current state of workspace",
        "Which service use our AI automation",
        "which models we provide training",
        "what is teachable-NLP"]
    },
    {
        "title":"DALL·E Abstract",
        "context" :"""Abstract 
            Text-to-image generation has traditionally focused on finding better modeling assumptions for training on a fixed dataset.
        These assumptions might involve complex architectures, auxiliary losses, or side information such as object part labels or segmentation masks supplied during training.
        We describe a simple approach for this task based on a transformer that autoregressively models the text and image tokens as a single stream of data.
        With sufficient data and scale, our approach is competitive with previous domain-specific models when evaluated in a zero-shot fashion.
        """,
        "questions":[
            "Where our model is competitive?"
        ]
    },
    {
        "title":"DALL·E Introduction 1",
        "context" :"""Introduction Modern machine learning approaches to text to image synthesis started with the work of Mansimov et al. (2015), who showed that the DRAW Gregor et al. (2015) generative model, when extended to condition on image captions, could also generate novel visual scenes. Reed et al. (2016b) later demonstrated that using a generative adversarial network (Goodfellow et al., 2014), rather than a recurrent variational auto-encoder, improved image fidelity. Reed et al. (2016b) showed that this system could not only generate objects with recognizable properties, but also could zero-shot generalize to held-out categories. Over the next few years, progress continued using a combination of methods. These include improving the generative model architecture with modifications like multi-scale generators (Zhang et al., 2017; 2018), integrating attention and auxiliary losses (Xu et al., 2018), and leveraging additional sources of conditioning information beyond just text (Reed et al., 2016a; Li et al., 2019; Koh et al., 2021). Separately, Nguyen et al. (2017) propose an energy-based framework for conditional image generation that obtained a large improvement in sample quality relative to contemporary methods. Their approach can incorporate pretrained discriminative models, and they show that it is capable of performing text-to-image generation when applied to a captioning model pretrained on MS-COCO. More recently, Cho et al. (2020) also propose a method that involves optimizing the input to a pretrained cross-modal masked language model. While significant increases in visual fidelity have occurred as a result of the work since Mansimov et al. (2015), samples can still suffer from severe artifacts such as object distortion, illogical object placement, or unnatural blending of foreground and background elements. Recent advances fueled by large-scale generative models suggest a possible route for further improvements. Specifically, when compute, model size, and data are scaled carefully, autoregressive transformers (Vaswani et al., 2017) have achieved impressive results in several domains such as text (Radford et al., 2019), images (Chen et al., 2020), and audio (Dhariwal et al., 2020). By comparison, text-to-image generation has typically been evaluated on relatively small datasets such as MS-COCO and CUB-200 (Welinder et al., 2010).""",
        "questions":[
            "which is previous research?",
            "which is the latest previous research?"
        ]
    },
    {
        "title":"DALL·E Introduction 2",
        "context" :"""Could dataset size and model size be the limiting factor of current approaches? In this work, we demonstrate that training a 12-billion parameter autoregressive transformer on 250 million image-text pairs collected from the internet results in a flexible, high fidelity generative model of images controllable through natural language. The resulting system achieves high quality image generation on the popular MS-COCO dataset zero-shot, without using any of the training labels. It is preferred over prior work trained on the dataset by human evaluators 90% of the time. We also find that it is able to perform complex tasks such as image-to-image translation at a rudimentary level. This previously required custom approaches (Isola et al., 2017), rather emerging as a capability of a single, large generative model.""",
        "questions":[
            "where the dataset come from?",
            "where the dataset which we use come from?"
        ]
    },
    {
        "title":"DALL·E Method 1",
        "context" :"""2. Method Our goal is to train a transformer (Vaswani et al., 2017) to autoregressively model the text and image tokens as a single stream of data. However, using pixels directly as image tokens would require an inordinate amount of memory for high-resolution images. Likelihood objectives tend to prioritize modeling short-range dependencies between pixels (Salimans et al., 2017), so much of the modeling capacity would be spent capturing high-frequency details instead of the low-frequency structure that makes objects visually recognizable to us. We address these issues by using a two-stage training procedure, similar to (Oord et al., 2017; Razavi et al., 2019): • Stage 1. We train a discrete variational autoencoder (dVAE)1 to compress each 256×256 RGB image into a 32 × 32 grid of image tokens, each element of which can assume 8192 possible values. This reduces the context size of the transformer by a factor of 192 without a large degradation in visual quality (see Figure 1). • Stage 2. We concatenate up to 256 BPE-encoded text tokens with the 32 × 32 = 1024 image tokens, and train an autoregressive transformer to model the joint distribution over the text and image tokens.""",
        "questions":[
            "what is first stage our training procedure ",
            "How many stages we have for training "
        ]
    },
    {
        "title":"DALL·E Method 2",
        "context" :"""The overall procedure can be viewed as maximizing the evidence lower bound (ELB) (Kingma & Welling, 2013; Rezende et al., 2014) on the joint likelihood of the model distribution over images x, captions y, and the tokens z for the encoded RGB image. We model this distribution using the factorization pθ,ψ(x, y, z) = pθ(x | y, z)pψ(y, z), which yields the lower bound ln pθ,ψ(x, y) > E z∼qφ(z | x) ln pθ(x | y, z) − β DKL(qφ(y, z | x), pψ(y, z)) , (1) where: • qφ denotes the distribution over the 32 × 32 image tokens generated by the dVAE encoder given the RGB image x 2 ; • pθ denotes the distribution over the RGB images generated by the dVAE decoder given the image tokens; and • pψ denotes the joint distribution over the text and image tokens modeled by the transformer. Note that the bound only holds for β = 1, while in practice we find it helpful to use larger values (Higgins et al., 2016). The following subsections describe both stages in further detail.""",
        "questions":[
            "what is perpose of our procedure",
            "Why we use β = 1?"
        ]
    },
]

streamlit.title('Find answer in the context')
QPR = streamlit.experimental_get_query_params()
#print(QPR)
API_URL = QPR["modelUrl"][0]

#API_URL = streamlit.sidebar.text_input("API URL", help = "API URL build from ainize.ai/teachable-nlp",
#                                        value = "https://train-1wbuxrd77ywfldccdhp0-gpt2-train-teachable-ainize.endpoint.dev.ainize.ai/predictions/deberta-en-base-pretrained-finetune")


use_sample_context = streamlit.sidebar.checkbox("use sample context", value = True)
if use_sample_context:
    use_sample_question = streamlit.sidebar.checkbox("use sample question", value = True)
else: use_sample_question = False

if use_sample_context:
    sample_id = streamlit.sidebar.selectbox("sample context", list(range(len(samples))), format_func=(lambda idx: samples[idx]["title"]), index = 0)
    title = samples[sample_id]["title"]
    default_context = samples[sample_id]['context']
else:
    sample_id = 0
    default_context = None


if use_sample_question:
    default_question = streamlit.sidebar.selectbox("sample questions", samples[sample_id]['questions'], index = 0)
else: default_question = None

inference.raw_based_inference(API_URL, default_question, default_context )