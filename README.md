# 日本語版 Pheme 音声合成モデル

このリポジトリは、音声合成モデル Pheme (https://github.com/PolyAI-LDN/pheme) を日本語向けにカスタマイズしたものです。Pheme の詳細については、元論文 [Pheme: Efficient and Conversational Speech Generation](https://arxiv.org/pdf/2401.02839.pdf) をご覧ください。

本リポジトリは、「ローカル LLM に向き合う会」と「メタデータラボ株式会社」の共催で開催された、[LOCAL AI HACKATHON#000](https://imminent-land-e64.notion.site/LOCAL-AI-HACKATHON-b8045ad0a99d40aaaa8591e41c5a6660) の成果物です。

本リポジトリのライセンスは Pheme 公式リポジトリと同じですが、[Hugging Face で公開している音声モデル](https://huggingface.co/offtoung/pheme-ja) (以下、本モデルと表記) については、下記の利用条件を守ってご利用ください。

## モデルの利用条件
**本モデルで生成した音声を公開する場合には使用した声のクレジットを記載してください。**

現在使用可能な声は、[黄琴海月さん](https://kikyohiroto1227.wixsite.com/kikoto-utau/kurage) と [ルナイトネイルさん](https://runaitoneiru.fanbox.cc/posts/3786422) です。

本モデルは、下記の禁止事項に該当する行為を除き、自由にご利用いただけます。

**禁止事項:**

・犯罪目的、差別目的、誹謗中傷目的、その他社会通念上不適切な目的で利用する行為

# 環境セットアップ
リポジトリのクローン
```
git clone https://github.com/offtoung/pheme-ja.git
```

conda 環境の作成:

``` 
conda create --name pheme3 python=3.10
conda activate pheme3

pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt --no-deps
pip3 install mecab-python3 unidic-lite jaconv
```
事前学習済みの T2S および S2A モデル、辞書、話者特徴量のダウンロード:

``` bash
git clone https://huggingface.co/offtoung/pheme-ja
mv pheme-ja/* .
```

事前学習済みの SpeechTokenizer のダウンロード

``` bash
st_dir="ckpt/speechtokenizer/"
mkdir -p ${st_dir}
cd ${st_dir}
wget "https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt"
wget "https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/config.json" 
cd ../../
```


# 音声の生成

次のようなコマンドを実行することで、音声が生成されます。
この例では、生成音声は output というディレクトリに格納されます。

```
python transformer_infer.py \
--text こんにちは、お元気ですか？ \
--outputdir output \
--voice runaitoneiru \
```

現在、話者として指定可能なのは、
- ルナイトネイルさん (runaitoneiru)
- 黄琴海月さん ひそひそスタイル (kikoto_kurage_hisohiso)
の 2 名です。

# 学習の実行方法
**※話者の追加希望があれば、[@offtoung](https://twitter.com/offtoung) までご連絡ください。ご本人の許諾が得られている音声については、積極的に追加して公開します。**

本リポジトリのモデルは、Pheme 公式リポジトリ (https://github.com/PolyAI-LDN/pheme) と互換性があります。
そのため、学習には、Pheme 公式リポジトリを用いることをおすすめします。

注意点は下記のとおりです。

・T2S モデル自体は互換性がありますが、前処理時の音素列への変換 (Phonemizer) には、本リポジトリの transformer_infer.py に含まれる Phonemizer を用いる必要があります。

・S2A モデルの学習は、音声ファイルのみで可能で、書き起こしは必要ありません。

・学習して結果得られたモデルで推論する際には、公式リポジトリを用いることがおすすめです。transformers_infer.py 内の PhemeClient.phonemizer を、本リポジトリの transformers_infer.py 内の Phonemizer に置き換えれば推論ができます。

・もし、学習して得られたモデルを本リポジトリのコードで推論する場合は、voice ディレクトリの直下に話者名のディレクトリを作成し、その直下に、(どれか一つの音声ファイルに対応する) 次の四つのファイルを保存してください。

- acoustic.npy: 前処理において audios-speechtokenizer/acoustic に作成されるファイル
- semantic.npy: 前処理において audios-speechtokenizer/semantic に作成されるファイル
- spkr_emb.npy: pyannote.embedding で得た話者埋め込み
- transcript.txt: 音声の書き起こし
  
これについては、Hugging Face のリポジトリ (https://huggingface.co/offtoung/pheme-ja/tree/main) を見ていただくのが分かりやすいと思います。

## Citation

本リポジトリのコードやモデルを利用した発表などを行う場合は、元論文を引用してください。

```Tex
@misc{budzianowski2024pheme,
      title={Pheme: Efficient and Conversational Speech Generation}, 
      author={Paweł Budzianowski and Taras Sereda and Tomasz Cichy and Ivan Vulić},
      year={2024},
      eprint={2401.02839},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
