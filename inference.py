
from time import gmtime, strftime
from helper import *

class IntegratedInference(object):
    def __init__(self):
        self.SENTIMENT_CONVERTER_PATH = SENTIMENT_CONVERTER_PATH
        self.SENTIMENT_TOKENIZER_PATH = SENTIMENT_TOKENIZER_PATH

        self.CHAT_BOT_MODEL_PATH = CHAT_BOT_MODEL_PATH
        self.CHAT_BOT_ENCODER_PATH = CHAT_BOT_ENCODER_PATH
        self.CHAT_BOT_TOKENIZER_PATH = CHAT_BOT_TOKENIZER_PATH

    def load_inference_models_objects(self):

        # Load Pickle Objects for Inference
        with open(self.SENTIMENT_TOKENIZER_PATH, 'rb') as fp:
            self.sentiment_tokenizer = pickle.load(fp)

        with open(self.CHAT_BOT_TOKENIZER_PATH, 'rb') as fp:
            self.chat_bot_tokenizer = pickle.load(fp)

        with open(self.CHAT_BOT_ENCODER_PATH, 'rb') as fp:
            self.encoder = pickle.load(fp)

        with open(CHAT_BOT_CONTENT_JSON) as content:
            data = json.load(content)

        self.responses = {intent['tag'] : intent['responses'] for intent in data['intents']}

        # Load the TFLite model and allocate tensors For Suicidal Early Detection Model
        interpreter = tf.lite.Interpreter(model_path=self.SENTIMENT_CONVERTER_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details

        # Load the H5 Models For Chat Bot Model
        self.model = tf.keras.models.load_model(self.CHAT_BOT_MODEL_PATH)

    @staticmethod
    def pad_tokens(post_tokens, 
                   max_length, 
                   padding='pre', 
                   truncating='pre'):
        post_tokens = post_tokens[0]
        if len(post_tokens) > max_length:
            if truncating == 'pre':
                post_tokens = post_tokens[-max_length:]
            elif truncating == 'post':
                post_tokens = post_tokens[:max_length]

        else:
            if padding == 'pre':
                post_tokens = [0] * (max_length - len(post_tokens)) + post_tokens
            elif padding == 'post':
                post_tokens = post_tokens + [0] * (max_length - len(post_tokens))
        return np.array([post_tokens])

    def SuicidalInference(self, text, user_name):
        original_text = text
        text = preprocess_one(text)
        post_tokens = self.sentiment_tokenizer.texts_to_sequences([text])
        
        post_pad_tokens = IntegratedInference.pad_tokens(
                                                        post_tokens, 
                                                        SENTIMENT_MAX_LENGTH, 
                                                        padding='pre', 
                                                        truncating='pre'
                                                        )

        post_pad_tokens = post_pad_tokens.astype(np.float32)
        input_shape = self.input_details[0]['shape']

        assert np.array_equal(
                        input_shape, 
                        post_pad_tokens.shape
                        ), "Expected Tensor Shape : {} but recieved {}".format(
                                                                            input_shape, 
                                                                            post_pad_tokens.shape
                                                                            )
        
        self.interpreter.set_tensor(self.input_details[0]['index'], post_pad_tokens)
        self.interpreter.invoke() # set the inference

        pred = self.interpreter.get_tensor(self.output_details[0]['index']) # Get predictions
        pred = pred.squeeze()
        sentiment = 'Non Suicidal' if pred > 0.5 else 'Suicidal'

        suicidal_response = {
                        "text": original_text,
                        "sentiment": sentiment,
                        "user_name": user_name
                            }

        return suicidal_response

    def BotInference(self, chat, user_name):
        original_chat = chat
        chat = preprocess_one(chat)
        chat_tokens = self.chat_bot_tokenizer.texts_to_sequences([chat])

        chat_pad_tokens = IntegratedInference.pad_tokens(
                                                        chat_tokens, 
                                                        CHAT_BOT_MAX_LENGTH, 
                                                        padding='pre', 
                                                        truncating='pre'
                                                        )
        chat_pad_tokens = chat_pad_tokens.astype(np.float32)

        #getting output from model
        output = self.model.predict(chat_pad_tokens)
        output = output.argmax()
        
        #finding the right tag and predicting
        response_tag = self.encoder.inverse_transform([output])[0]
        HelpingBot = np.random.choice(self.responses[response_tag])

        time_stamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        bot_response = {
                 "user" : original_chat,
                 "bot": HelpingBot,
                 "user_name": user_name,
                 "time_stamp": time_stamp
                 }

        return bot_response
