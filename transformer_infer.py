"""Inference logic.

Copyright PolyAI Limited.
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from einops import rearrange
from librosa.util import normalize
from pyannote.audio import Inference
from transformers import GenerationConfig, T5ForConditionalGeneration

import constants as c
from data.collation import get_text_semantic_token_collater
from data.semantic_dataset import TextTokenizer
from modules.s2a_model import Pheme
from modules.vocoder import VocoderType

import re
import MeCab
import jaconv

kana_symbols = ['ァ', 'ア', 'ィ', 'イ', 'ゥ', 'ウ', 'ェ', 'エ', 'ォ', 'オ', 'カ', 'ガ', 'キ', 'ギ', 'ク', 'グ', 'ケ', 'ゲ', 'コ', 'ゴ', 'サ', 'ザ', 'シ', 'ジ', 'ス', 'ズ', 'セ', 'ゼ', 'ソ', 'ゾ', 'タ', 'ダ', 'チ', 'ッ', 'ツ', 'テ', 'デ', 'ト', 'ド', 'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'バ', 'パ', 'ヒ', 'ビ', 'ピ', 'フ', 'ブ', 'プ', 'ヘ', 'ベ', 'ペ', 'ホ', 'ボ', 'ポ', 'マ', 'ミ', 'ム', 'メ', 'モ', 'ャ', 'ヤ', 'ュ', 'ユ', 'ョ', 'ヨ', 'ラ', 'リ', 'ル', 'レ', 'ロ', 'ワ', 'ン', 'ヴ'] 

pmarks = ['_', '！', '？', '!', '?', ']', '[', '。', '、', 'ー', ' ']
acceptable_symbols = set(kana_symbols + pmarks)

class Phonemizer(): 
  def __init__(self): 
   self.tagger = tagger = MeCab.Tagger("-d unidic-tdmelodic")

  def __call__(self, text):
    text = re.sub(r"(「|」|（|）|｛|｝|【|】|『|』|［|］|＜|＞|《|》|〈|〉|\(|\)|[|]|{|}|<|>)", "", text)

    text = text.replace("…", "")
    s = jaconv.normalize(text, "NFKC").strip()

    m = self.tagger.parse(s).splitlines()[:-1]
    kana = ''
    was_pmark = False
    for idx, elem in enumerate(m):
        if '\t' not in elem: 
            continue
        cols = elem.split('\t')
        pmark = "補助記号" in cols[4]

        if pmark:
          kana += cols[3]
          was_pmark = True
        else:
          if idx > 0 and not was_pmark:
            kana += "_" 
           
          was_pmark = False

          yomi = cols[1].split(',')[-1]
          if yomi == '*': 
            yomi = ""
          kana += yomi

    kana = jaconv.hira2kata(kana)


    kana = jaconv.z2h(kana, kana=False, digit=True, ascii=True)

    kana = kana.replace("ヴァ", "バ").replace("ヴィ", "ビ").replace("ヴェ", "ベ").replace("ヴォ", "ボ").replace("ヴ", "ブ").replace('ヂ', 'ジ').replace('ヅ', 'ズ').replace('ヂ', 'ジ').replace('ヮ', 'ワ').replace('ヱ', 'エ').replace('ヲ', 'オ').replace('・', '').replace("]ー", "ー]").replace("[ー", "ー[")

    kana = "".join([x for x in list(kana) if x in acceptable_symbols])

    hira = jaconv.kata2hira(kana)
    jl = jaconv.hiragana2julius(hira)

    jl = jl.replace("ゃ", " y a ")
    jl = jl.replace("ゅ", " y u ")
    jl = jl.replace("ょ", " y o ")
    jl = jl.replace("ぁ", " a ")
    jl = jl.replace("ぃ", " i ")
    jl = jl.replace("ぅ", " u ")
    jl = jl.replace("ぇ", " e ")
    jl = jl.replace("ぉ", " o ")

    jl = jl.replace("]", "").replace("[", "")
    jl = jl.replace(" ", "")
    jl = jl.replace("、", ",_").replace("。", ".")

    pos = jl.find(":")
    while pos != -1:
      if pos > 0:
        jl = jl[:pos]+jl[pos-1]+jl[pos+1:]
      else:
        jl = jl[1:]
      pos = jl.find(":") 

    #jl = jl.replace("_", " ")
    return list(jl)

# How many times one token can be generated
MAX_TOKEN_COUNT = 100

logging.basicConfig(level=logging.DEBUG)
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text", type=str,
        default="こんにちは。お元気ですか？"
    )
    parser.add_argument("--outputdir", type=str, default="output/")
    parser.add_argument(
        "--text_tokens_file", type=str,
        default="ckpt/unique_text_tokens.k2symbols"
    )
    parser.add_argument("--t2s_path", type=str, default="ckpt/t2s/")
    parser.add_argument(
        "--s2a_path", type=str, default="ckpt/s2a/s2a.ckpt")

    parser.add_argument("--target_sample_rate", type=int, default=16_000)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=210)
    parser.add_argument("--voice", type=str, default="runaitoneiru")

    return parser.parse_args()


class PhemeClient():
    def __init__(self, args):
        self.args = args
        self.outputdir = args.outputdir
        self.target_sample_rate = args.target_sample_rate
        self.collater = get_text_semantic_token_collater(args.text_tokens_file)
        #self.phonemizer = TextTokenizer()
        self.phonemizer = Phonemizer()
    
        # T2S model
        self.t2s = T5ForConditionalGeneration.from_pretrained(args.t2s_path)
        self.t2s.to(device)
        self.t2s.eval()

        # S2A model
        self.s2a = Pheme.load_from_checkpoint(args.s2a_path)
        self.s2a.to(device=device)
        self.s2a.eval()

        # Vocoder
        vocoder = VocoderType["SPEECHTOKENIZER"].get_vocoder(None, None)
        self.vocoder = vocoder.to(device)
        self.vocoder.eval()

        self.spkr_embedding = "voice" / Path(args.voice) / "spkr_emb.npy"
        self.acoustic_prompt = np.load("voice"/Path(args.voice)/"acoustic.npy").squeeze().T
        self.semantic_prompt = np.load("voice"/Path(args.voice)/"semantic.npy")
        with open("voice" / Path(args.voice) / "transcript.txt", "r", encoding="utf-8") as f:
          self.prompt_text = f.readline()

    def lazy_decode(self, decoder_output, symbol_table):
        semantic_tokens = map(lambda x: symbol_table[x], decoder_output)
        semantic_tokens = [int(x) for x in semantic_tokens if x.isdigit()]

        return np.array(semantic_tokens)

    def infer_text(self, text, voice, sampling_config):
        semantic_prompt = np.load("voice" / Path(self.args.voice) / "semantic.npy")  # noqa
        phones_seq = self.phonemizer(text)
        input_ids = self.collater([phones_seq])
        input_ids = input_ids.type(torch.IntTensor).to(device)

        labels = [str(lbl) for lbl in semantic_prompt]
        labels = self.collater([labels])[:, :-1]
        decoder_input_ids = labels.to(device).long()
        #logging.debug(f"decoder_input_ids: {decoder_input_ids}")

        counts = 1E10
        while (counts > MAX_TOKEN_COUNT):
            output_ids = self.t2s.generate(
                input_ids, decoder_input_ids=decoder_input_ids,
                generation_config=sampling_config).sequences
            
            # check repetitiveness
            _, counts = torch.unique_consecutive(output_ids, return_counts=True)
            counts = max(counts).item()

        output_semantic = self.lazy_decode(
            output_ids[0], self.collater.idx2token)

        # remove the prompt
        return output_semantic[len(semantic_prompt):].reshape(1, -1)

    def infer_acoustic(self, output_semantic, voice):
        semantic_tokens = output_semantic.reshape(1, -1)
        acoustic_tokens = np.full(
            [semantic_tokens.shape[1], 7], fill_value=c.PAD)

        # Prepend prompt
        acoustic_tokens = np.concatenate(
            [self.acoustic_prompt, acoustic_tokens], axis=0)
        semantic_tokens = np.concatenate([
            self.semantic_prompt[None], semantic_tokens], axis=1)

        # Add speaker
        acoustic_tokens = np.pad(
            acoustic_tokens, [[1, 0], [0, 0]], constant_values=c.SPKR_1)
        semantic_tokens = np.pad(
            semantic_tokens, [[0,0], [1, 0]], constant_values=c.SPKR_1)

        speaker_emb = None
        if self.s2a.hp.use_spkr_emb:
            speaker_emb = np.repeat(
                self.speaker_emb, semantic_tokens.shape[1], axis=0)
            speaker_emb = torch.from_numpy(speaker_emb).to(device)
        else:
            speaker_emb = None

        acoustic_tokens = torch.from_numpy(
            acoustic_tokens).unsqueeze(0).to(device).long()
        semantic_tokens = torch.from_numpy(semantic_tokens).to(device).long()
        start_t = torch.tensor(
            [self.acoustic_prompt.shape[0]], dtype=torch.long, device=device)
        length = torch.tensor([
            semantic_tokens.shape[1]], dtype=torch.long, device=device)

        codes = self.s2a.model.inference(
            acoustic_tokens,
            semantic_tokens,
            start_t=start_t,
            length=length,
            maskgit_inference=True,
            speaker_emb=speaker_emb
        )

        # Remove the prompt
        synth_codes = codes[:, :, start_t:]
        synth_codes = rearrange(synth_codes, "b c t -> c b t")

        return synth_codes

    def generate_audio(self, text, voice, sampling_config):
        start_time = time.time()
        output_semantic = self.infer_text(
            text, voice, sampling_config
        )
        logging.debug(f"semantic_tokens: {time.time() - start_time}")

        start_time = time.time()
        codes = self.infer_acoustic(output_semantic, voice)
        logging.debug(f"acoustic_tokens: {time.time() - start_time}")

        start_time = time.time()
        audio_array = self.vocoder.decode(codes)
        audio_array = rearrange(audio_array, "1 1 T -> T").cpu().numpy()
        logging.debug(f"vocoder time: {time.time() - start_time}")

        return audio_array

    @torch.no_grad()
    def infer(
        self, text, voice="male_voice", temperature=0.7,
        top_k=210, max_new_tokens=1200,
    ):
        sampling_config = GenerationConfig.from_pretrained(
            self.args.t2s_path,
            top_k=top_k,
            num_beams=1,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True
        )

        text = self.prompt_text + " " + text

        audio_array = self.generate_audio(text, voice, sampling_config)
        return audio_array


if __name__ == "__main__":
    args = parse_arguments()
    args.outputdir = Path(args.outputdir).expanduser()
    args.outputdir.mkdir(parents=True, exist_ok=True)

    client = PhemeClient(args)
    audio_array = client.infer(args.text, voice=args.voice)
    sf.write(os.path.join(
        args.outputdir, f"out.flac"), audio_array, 
        args.target_sample_rate
    )
