seed = 42

database = 'HelpingHand'
suicidal_collection = 'suicidal_prediction'
chat_bot_collection = 'chat_bot_prediction'

db_url = "mongodb+srv://admin:admin@early-detection.jevtz.mongodb.net/test"

aws_url = '0.0.0.0'
aws_port = 8080

SENTIMENT_MAX_LENGTH = 120
CHAT_BOT_MAX_LENGTH = 5
oov_token = '<OOV>'

SENTIMENT_CONVERTER_PATH = 'src/early_detection.tflite'
SENTIMENT_TOKENIZER_PATH = 'src/SENTIMENT_TOKENIZER.pkl'

CHAT_BOT_ENCODER_PATH = 'src/TAG_ENCODER.pkl'
CHAT_BOT_CONTENT_JSON = 'src/content.json'
CHAT_BOT_MODEL_PATH = 'src/SUICIDE-BOT.h5'
CHAT_BOT_TOKENIZER_PATH = 'src/CHAT_BOT_TOKENIZER.pkl'