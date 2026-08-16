import torch
import sentencepiece as spm
from translator import Seq2Seq, Encoder, Decoder, Attention 

# ---------------------#
# 1. Load Tokenizers
# ---------------------#
sp_en = spm.SentencePieceProcessor()
sp_en.load('en_bpe.model')
sp_cn = spm.SentencePieceProcessor()
sp_cn.load('zh_bpe.model')
PAD_ID = sp_en.pad_id()
BOS_ID = sp_en.bos_id()
EOS_ID = sp_en.eos_id()

# ---------------------#
# 2. Load Trained Model
# ---------------------#
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters (must match training)
INPUT_DIM = sp_en.get_piece_size()
OUTPUT_DIM = sp_cn.get_piece_size()
ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
HID_DIM = 512
N_LAYERS = 3

attention = Attention(HID_DIM).to(DEVICE)
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, n_layers=N_LAYERS).to(DEVICE)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attention, n_layers=N_LAYERS).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
model.load_state_dict(torch.load('seq2seq_bpe_attention.pt', map_location=DEVICE))
model.eval()

# ---------------------#
# 3. Translation Function (Greedy)
# ---------------------#
def translate_sentence(sentence, max_len=100):
    # Tokenize and convert to IDs
    tokens = [BOS_ID] + sp_en.encode(sentence, out_type=int) + [EOS_ID]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(DEVICE)  # [src_len, 1]
    src_len = [len(tokens)]

    # 调用Encoder
    with torch.no_grad():
        encoder_outputs, hidden, cell = encoder(src_tensor, src_len)

    # 第一个输入token，序列起始token：<bos>
    trg_indices = [BOS_ID]
    # 逐个token，循环调用Decoder。
    for _ in range(max_len):
        # 最新生成的token作为输入
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(DEVICE)
        with torch.no_grad():
            output, hidden, cell, _ = decoder(trg_tensor, hidden, cell, encoder_outputs,
                                               (src_tensor != PAD_ID).permute(1, 0))
        # 取预测概率最大的token作为输出
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)
        if pred_token == EOS_ID:
            break

    # 将token id解码为文字 (跳过<bos>和<eos>)
    translated = sp_cn.decode(trg_indices[1:-1])
    return translated

# ---------------------#
# 4. Interactive Test
# ---------------------#
if __name__ == '__main__':
    while True:
        src_sent = input("Enter English sentence (or 'quit' to exit): ")
        if src_sent.lower() in ['quit', 'exit']:
            break
        translation = translate_sentence(src_sent)
        print(f"Chinese Translation: {translation}\n")