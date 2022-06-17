import tokenizers
import transformers

MAX_LEN = 170
DEVICE = "cuda"
TRAIN_BATCH_SIZE = 7
VALID_BATCH_SIZE = 4
EPOCHS = 15
BERT_PATH='SAV'
MODEL_PATH = "weights/tweet_model.bin"
TRAINING_FILE = "data/train.csv"
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    f"vocab.txt", 
    lowercase=True
)
