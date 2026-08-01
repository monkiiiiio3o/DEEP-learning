#构建NLP词典
import sentencepiece as spm
spm.SentencePieceTrainer.Train('--input="C:\\Users\\Monki\\Desktop\\deep-learing\\en2cn\\train_en.txt" --model_prefix=en_bpe --vocab_size=16000 --model_type=bpe --character_coverage=1.0 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3')
spm.SentencePieceTrainer.Train('--input="C:\\Users\\Monki\\Desktop\\deep-learing\\en2cn\\train_zh.txt" --model_prefix=zh_bpe --vocab_size=16000 --model_type=bpe --character_coverage=0.9995 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3')