""" Encoder configurations """
MDL_ENCODER = 'encoder'  #: ENCODER KEY
#DEF_ENCODER = "microsoft/cocolm-base"
DEF_ENCODER = 'google/electra-base-discriminator'  #: DEFAULT ENCODER VALUE

""" Decoder configurations """
MDL_EQUATION = 'equation'  #: DECODER KEY
MDL_Q_PATH = 'path'  #: PATH TO LOAD DECODER
MDL_Q_ENC = 'encoder_config'  #: ENCODER CONFIGURATION
MDL_Q_EMBED = 'embedding_dim'  #: EMBEDDING DIMENSION
MDL_Q_HIDDEN = 'hidden_dim'  #: HIDDEN STATE DIMENSION
MDL_Q_INTER = 'intermediate_dim'  #: INTERMEDIATE LAYER DIMENSION
MDL_Q_LAYER = 'layer'  #: NUMBER OF HIDDEN LAYERS
MDL_Q_INIT = 'init_factor'  #: INITIALIZATION FACTOR FOR TRANSFORMER MODELS
MDL_Q_LN_EPS = 'layernorm_eps'  #: EPSILON VALUE FOR LAYER NORMALIZATION
MDL_Q_HEAD = 'head'  #: NUMBER OF MULTI-ATTENTION HEADS

""" Explainer configurations """
MDL_EXPLANATION = 'explanation'  #: EXPLAINER KEY
MDL_X_SHUFFLE_ON_TRAIN = 'shuffle' 
MDL_X_R_BOTH = -1  #: Recombine as SWAN
MDL_X_R_SUFF = 0  #: Enforce sufficiency (text 0%)
MDL_X_R_COMP = 1  #: Enforce comprehensiveness (text 100%)

""" Decoder configuration default """
DEF_Q_EMBED = 128  #: FALLBACK VALUE FOR EMBEDDING DIM
DEF_Q_HIDDEN = 768  #: FALLBACK VALUE FOR HIDDEN DIM
DEF_Q_INTER = 2048  #: FALLBACK VALUE FOR INTERMEDIATE DIM
DEF_Q_LAYER = 6  #: FALLBACK VALUE FOR NUMBER OF HIDDEN LAYERS
DEF_Q_INIT = 0.02  #: FALLBACK VALUE FOR INITIALIZATION FACTOR
DEF_Q_LN_EPS = 1E-8  #: FALLBACK VALUE FOR LAYER NORMALIZATION EPSILON
DEF_Q_HEAD = 12  #: FALLBACK VALUE FOR NUMBER OF MULTI-HEAD ATTENTIONS
