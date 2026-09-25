# deep-learning-from-scratch-6

## 0章

- xii. encoder-decoderでの翻訳を考えた時、encoderでは入力文を最初に1回処理するのみ。以降の翻訳文を生成する過程ではdecoderのみで再帰的に処理する。勘違いしやすいところ。具体的には、encoder出力は固定で、decoderがattentionで毎ステップ参照する。
- xv. tokenizerとmodelは別々に学習される。tokenizerの学習はNNではなく統計的手法(BPE等)で行う。tokenizerの品質も最終的なLLMの性能に大きな影響を与える。
- xvi. scaling lawsの図。縦軸横軸はともに対数スケールで表されており、これが直線になるということはべき乗則が成り立つということ。
    - 計算量C, データサイズD, パラメタ数Nに対して計測。
    - 縦軸はTest Loss。（学習したモデルに対してテストデータでLossを計測したもの）
    - べき乗則は L∝N^(−0.076)のように表される。例えばパラメタ数Nを10倍にしたら、(L_増加後/L_増加前)=10^(-0.076)=0.84..となる。つまりLossは0.84倍となる。
    - C, D, Nは何れも資源を10倍にしたとき、Test Lossは0.8~0.89倍ほどになる（11~20%ほど下がる）
    - Scaling Lawsでの重要な点は２つ
        - 「規模を大きくすれば改善する」
        - 「同じ改善を続けるには資源を掛け算で増やす必要がある」（計算量Cを1日から10日に増やすのは簡単だが、1000日から10000日に増やすのは難しい。資源をずっと増やし続けることはそう簡単なことではない。）
    - 計算量Cの図だけたくさんの青線があるのは？→異なるサイズのモデルを学習させた場合の結果。オレンジの破線は計算量ごとに上手くモデルサイズや学習量を選んだ時の到達ライン。
    - C, D, Nはバランスよく増やす必要がある。どれかだけ高くてもダメ。
    - 3つの図において、それぞれ対象とする変数以外の２変数は固定しているわけではなく、制約とならないように調整している。例えば、Cの図を作る場合、Dは十分に用意しNは予算に応じて調整する。Dの図を作る場合、モデルは十分に大きくして(N)、過学習を避けるために早期停止する(C)。Nの図を作る場合、データを十分に用意し(D)、各モデルが収束するまで学習させる(C)。つまり、Cの図は「計算予算による限界」、Dの図は「データ不足による限界」、Nの図は「モデルサイズによる限界」を表している。
    - そもそも計算量C(Compute)は他の値と独立ではない。C=6NBSと近似される。N:パラメタサイズ、B:1stepあたりのトークン数、S:学習ステップ数。BS=T(累積処理トークン数)と見なされる。データサイズD=1億で3epoch学習される場合、T=3億となる。
    - べき乗則についての補足：べき乗則 y=kx^a の両辺の対数を取ると、log y = logk + a*logx となる。log yを縦軸、log xを横軸とすると、aがマイナスの値になるので、図が直線になる意味がわかる。
- xix. SFTではChatML(Markup Language)形式が利用される
    - <|im_start|>, <|im_sep|>, <|im_end|>という特殊トークンを利用。
    - 例：<|im_start|> user <|im_sep|> こんにちは <|im_end|> <|im_start|> assistant <|im_sep|> こんにちは！ <|im_end|>
    - SFT時の学習ロジックは事前学習と同じNext Token Prediction。
- xxii. ChatGPTのRLHFでよく出てくる図は左から順に以下を表している。
    - SFT
    - Reward Modelの学習
    - Reward Modelを使ったPPO
- xxiii. DeepSeek-R1-Zero (2025)はSFTをしていない。高性能なベースモデルに強化学習のみを適用。報酬は<think>タグと<answer>タグの使い方が正しいか、<answer>タグ内の解答が正しいかで計算。DeepSeek-R1-Zeroはthinkの中で解き方を独自に見つけることから、AlphaGo Zero(2017)を想起させる。

## 1章 CodeBot Tokenizer

- p6. Unicode: ord('h')で104, chr(104)でhが出力。Unicodeは世界中の文字に番号を割り当てる国際標準規格で、番号をCode Pointという。
- p8. UTF-8エンコード: UTF-8はUnicode文字をバイト列に変換するエンコード方式。
    ```python
    encoded = 'あ'.encode("utf-8")
    print(encoded) # b'\xe3\x81\x82'
    print(list(encoded)) # [227, 129, 130]
    decoded = bytes([227, 129, 130]).decode('utf-8')
    print(decoded) # 'あ'
    ```
    - どんな文字でも最終的には0~255の数値列で表される。これにより語彙サイズを256にできるとも言える。
- p30. 事前トークン化: 句読点などは事前にsplitされるようにregexで処:wq
理する。これにより、' hello.'や' hello?'や' hello!'のような無駄なエンコードが増えずに済む。
- Byte Pair Encoding (Byte-level BPE) には未知語[UNK]が存在しない。UTF-8エンコードでは必ず0~255の数値に分解できるため、必ずどれかで表される。
- p40. 圧縮率: 1トークンが平均何バイトを表現するか
    ```python
    text = "(十分大きなテキスト)"
    byte_count = len(text.encode("utf-8"))
    token_count = tokenizer.encode(text)
    compression_ratio = byte_count / token_count
    ```
    - 例えばtextが10000 Bytes（英字10000文字とか）でtoken_countが4700の場合、圧縮率は10000/4700=2.13倍
    - 語彙サイズを増やせば圧縮率は高くなる。ただし、語彙サイズが大きいということは、モデルでnn.Embeddingで表現される学習パラメタが増えることを意味する。
    - GPT-4などではトークナイザとして`cl100k_base`(語彙サイズ100,277, 圧縮率3.97倍)が利用される

## 2章 CodeBot Model

- p56. Softmaxは「高い値は極端に高くする」という性質がある。例えば、Softmax([100, 200, 300])=[0.0000, 3.7835e-44, 1.0000e+00]となり、300がほぼ1になる。逆に類似度がまあまあかなくらいのものは全て0になる。
- p56. Softmaxの飽和：飽和とは入力を変化させても出力が変わらないこと。上に書いたような性質により、このようなことが起こり、学習に悪影響がある。そのため、スケーリング(softmaxの中の分母であるルートd)が必要。
- p86. AttentionにおけるCausal Maskの作り方。縦をquery, 横をkeyとした場合、右上の三角成分が未来に相当する情報であるため、-infとすることで、softmax出力を0にできる。実装は以下のようにtorch.triu(triangle upper) or torch.tril(triangle lower)を使って行う
    ```python
    # x: (Batch, Context length, Embed dim)
    # Q: (B, C, Key dim)
    # K: (B, C, K)
    # V: (B, E, E)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores /= self.key_dim ** 0.5

    # trilでやる場合
    mask = torch.tril(
        torch.ones(C, C)
    )
    scores = scores.masked_fill(
        mask == 0, float('-inf')
    )

    # triuでやる場合
    mask = torch.triu(
        torch.ones(C, C),
        diagonal=1
    )
    scores = scores.masked_fill(
        mask, float('-inf')
    )

    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    ```
- p88. Valueが最も重いので低ランク行列で表す。GPT-3ではembed_dim=12288, key_dim=128。つまり一般的なLLMではAttentionのVが最も大きい。12288x12288=約1.5億。なので行列分解で低ランク近似。すると、式変形によりAttention内部では完全にkey_dim=128のみの行列計算で表すことができ、その出力ベクトル output: (B, C, K)に対してW_o: (K, E)による線形変換を行えば、元のAttentionと同様に見做せる。
- p96. Q, K, Vの重みは分けずとも1つのLinearでより効率的にかける
    ```python
    Q = W_q(x)
    K = W_k(x)
    V = W_v(x)

    # 以下のようにもっと効率化できる
    W = nn.Linear(E, 3*H*D, bias=False)
    Y = W(x) # (B, C, 3*H*D)
    Q, K, V = Y.chunk(3, dim=2) # (B, C, H*D)が３つ
    ```
- p96. Multi-head Attentionはhead分だけ並列にAttentionを実行する仕組み。重みの初期値が異なるため、headごとに異なる特徴を学習するように収束する。仕組み: これまでのsingle headの計算ではkey_dimを自由に設定してきた。しかし、これをH*D(ヘッド数*ヘッド次元)とし、headごとに分けて計算する。(Dのヘッド次元がこれまでのkey_dimに該当し、それをH分だけ並列に行うイメージ。)ただし、ここは行列計算の行ごとに独立して計算するという特徴があるので、headごとに別々な重み(Linear)を定義する必要はなく、「共通のLinearで処理→viewで行列の形状を変化」とすることで実現している。
- p98. transpose, permuteなどの処理をするとテンソルの要素位置が不連続になる。view操作はメモリ配置は連続していることを前提とするため、viewの前にcontiguous()で整える。
- p106. FFNはいろいろな種類があるが、ここでは、Sequential(Linear, GELU, Linear, Dropout)を採用。中間表現の次元数は自由に設定できるが、一般的に「4倍」が最も性能が高いとされる。
- p107. Multi-head Attentionにおけるhead_dimは何にでも指定することはできるが、(embed_dim // n_head)の値を指定するのが一般的。これによりMultiHeadAttentionクラスが持つパラメタ数が一定に保たれる。
- p110. 重み共有: 最初のnn.Embedding(vocab_size, embed_dim)と、最後のnn.Linear(embed_dim, vocab_size)はやっていることが逆。そこで、これらは重みを共有する。具体的には`self.embed.weight = self.unembed.weight`の1行。
- p110. GPT2の初期値は、平均0、標準偏差0.02で行われる。

## 3章 CodeBot 学習

- p120. optimizerはAdamWが一般的。Lossは多クラス分類問題なのでcross_entropy。
- p120. 事前学習時、cross_entropy(logits, labels)で損失を計算する。つまり、学習時は最後にsoftmaxをする必要はない。推論時の単語サンプリング時に初めてsoftmaxを使う。
- p124. 温度T: `logits = logits / temperature`に使いsoftmaxをかける。Tは自由に値を取ることができ、1だと標準。0に近づくと決定的になり、大きくなるとランダム性が増す。
- p125. Top-kサンプリング: 確率分布の上位k個を候補とする。
- p126. Top-pサンプリング: 累積確率が閾値p（例: 0.9）に達するまでの単語を候補とする。
- p126. Top-kとTop-pの併用: まずは１つずつ試す方が良いが、より安定性をもたらすために併用することもある。その場合はTop-kを先にしてその次にTop-pを行う。その際、huggingfaceでは、Top-k後に確率総和が再度1になるように算出され、その上でTop-pを適用するのが一般的。
- p163. GRPOでは古い方策モデルを使ってデータをサンプリングし、その時にAdvantageも算出する。そのあとはそれぞれを１サンプルとして扱いバッチを作成する
- p163. GRPOのLossの算出：上記方法でOld modelをN(=8)回実行してそれぞれのAdvantageを算出している状況。その後の流れを説明する。１サンプルは（prompt, response, advantage）。modelは学習対象とOldの２つ使う。まずpromptとresponseを結合して１つの文字列を作成する。学習対象モデルとold modelのそれぞれにそれを入力し、その文字列の出力確率分布を得る。サイズは(Batch size, Context length)。（実際に２つのモデルに生成させているわけではなく、対象の文字列を入力した時の確率分布を得る）。これで２つのモデルそれぞれの出力確率分布が得られ、AdvantageもあるのでLossを算出できる。これ以降は、そのバッチ、かつそのトークン位置での計算。2つのモデルの出力確率からratioを求め、ratio * advantageが損失のベースとなる。ただし、大きすぎたり小さすぎたりしないようにclipする。その後、prompt位置であったりpadding位置であったりは学習しなくて良いのでそこはmaskする。attention maskの時みたいな便利なものはないので、普通に0, 1のマスクで掛け算する。最後に（Batch size, Context length)ごとに別々で求めていたものを合計し、サンプル数で割って負の符号をつけたら損失になる。

## 4章 StoryBot Tokenizer

- p182. BPE学習に必要なのは、全体に対するカウント(ids_counts)、トークンペアに対するカウント(pair_counts)。このままではペアをマージした時にids_countsの再算出が必要になってしまうので、トークンペアに対する関連する事前トークン列のキャッシュ(pair_to_ids)を導入する。
- p186. テキスト全部open + 処理だとメモリ消費が激しいので、100MBごとに処理する。
    ```python
    # 1000~1500バイトを読み込む処理
    with open(filepath, "rb")as f: # バイナリモードで読み込み
        f.seek(1000) # 1000バイトの位置にジャンプ
        chunk_bytes = f.read(500) # 500バイトを読み込み
    ```
    - しかし中途半端な位置で区切れてしまう可能性がある。そのため、<|endoftext|>のバイト位置を事前に特定しておく。それを1chunkと考えてNchunkごとに区切る。
- p199. 事前トークンの並列処理：multiprocessing の　Pool を使って並列処理。並列処理が効果的な場面としてCPUバウンドな処理とI/Oバウンドな処理があるが、今回はCPUバウンドと言える。

# 5章 StoryBot Model

- p214. RoPEは相対位置を考慮した位置エンコーディング。QとKをtokenの相対位置に基づいて回転させて位置情報を埋め込む。回転角度thetaは固定値10000とmax_context_lenとベクトル次元数から自然と決まる。ベクトルの回転は(x0, x1), (x2, x3), ...のようにベクトルを分解して２次元ベクトルに分解した上で適用する。RoPEは学習パラメタを持たない。
- p214. init関数でregister_buffer("cos_cache", cos)のようにモデルの一部として保持したいテンソルを登録しておくと、forward関数での計算の時にself.cos_cacheのように呼び出せる。
- p216. SwiGLU: Swish関数（x*sigmoid(x)）を使ってGated Linear Unitを構築したもの。GLUはLinearで分岐させ片方に活性化関数を通しそれらの要素積をとったものをさらにLinearに通した構成のFFN。活性化関数をReLU, GeLU, Swishのどれにするかで、REGLU, GEGLU, SwiGLUとよぶ。なぜSwiGLUが最も効果が高いかを理論的には説明できておらず、神のご加護と論文内に記載されている。
- p219. SwiGLUなどのゲート付きFFNでは、中間層の次元数は8/3にするのが一般的。GPT-2のときは入力サイズの4倍にしており、一般的にそれが経験的に高精度になるからであった。その際Linearは２つだった。一方で今回は、Linearが３つになっている。GPT-2の時と同じパラメタ数に揃えようとすると、(2*4)/3倍すれば良いことになる。
- p220. RMSNormはLayerNormを改良したもので、LayerNormで効果がなかった部分（平均値muの減算、学習パラメタbeta）を削除した形になっている。
- p222. そのほか、Dropoutの削除（LLMでは事前学習時のコーパスが大きいので、複数エポック回す必要がなく1回だけしか学習しないので、過学習が起きにくいため。）、Linearのバイアスの削除、重み共有の削除を行なっている。
- p227. KV cache: 推論時のみKV cacheを適用する。（学習時は勾配計算ができなくなってしまうので行わない。）KV cacheはPrefillとDecodeの２つの処理がある。最初に入力文を処理するのがPrefill。この時単純にKとVを計算し、k_cache, v_cacheとしてそれぞれ保持する。そして2token目以降では新たに入力される1tokenのみを処理し、そこで求めたK, Vでk_cache, v_cacheを拡張する。ちなみに、Qは1token分しか毎回処理しないので、cacheする必要がない。
- p236. RoPEとKV cacheを一緒に使う場合、位置エンコーディングの扱いには注意が必要。KV cacheを使うなら毎回1tokenしか入力されないので、RoPEで今何トークン目かを表す引数offsetが必要になる。


# 6章 StoryBot 学習

- p237. Momentum: SGDは層ごとに勾配のスケールが大きく異なるため学習が安定しないという問題点がある。そこで、勾配の指数移動平均を扱うことする。（アイデアとしては、勾配を毎回完全に新しく算出するのではなく、過去のステップも考慮して計算しようというもの）
- p238. Adam. 勾配の１次モーメントと２次モーメントを追跡し、パラメタごとに学習率を自動調整する。勾配の変動が大きいパラメタは小さく、勾配の変動が小さいパラメタは大きく更新する。（Adaptive moment estimation）
- p240. AdamにL2正則化を加えたものをAdamWという。通常のL2正則化ではLossにlambda/2*|theta^2|を加えれば良いが、Adamの場合は更新式が複雑なので、それではうまく正則化が適用されない。そこで、損失に加えるのではなく、Adamの計算時に正則化項を最後に追加する。
- p246. 学習率スケジューリング：最初は学習率を徐々に上げ（warmup）、その後徐々に学習率を下げる（Annealing: アニーリング）。これにより学習が安定する。
- p247. アニーリングの種類
    - Cosine Annealing: Cosine関数に従って滑らかに減衰させる手法。数式がやや複雑で、最小学習率の設定が必要。
    - Linear decay-to-zero (D2Z)：シンプルな方法。学習率を線型的にゼロまで減衰させる。最小学習率の設定も不要。
- p249. `from torch.optim.lr_scheduler import CosineAnnealingLR`のように呼び出せる。ただしD2Zのような新しい手法は未登録。
- p249. 混合精度：何も指定しなければFP32。FP16, BF16に変える事で効率化。
    - FP16: 指数部5bit, 仮数部10bit, 符号1bit。指数部が小さいので扱える範囲が狭いが、仮数部が大きいので有効数字が大きく細かな値まで扱える。
    - BF16: 指数部8bit, 仮数部7bit, 符号1bit。指数部が大きいので扱える範囲が広いが、仮数部が小さいので有効数字が小さく、大雑把な値しか扱えない。
    - オーバーフローやアンダーフローがおきやすいのはFP16。機械学習では扱える数字の範囲が重要なのでBF16の方が頻繁に利用される。ただし、BF16でも桁落ちの問題は残る。そこで、精度を使い分ける混合精度が利用される。
    ```python
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        b = a @ a # 行列積はBF16
        c = a.sum() # 累積はFP32
    ```
    - 実際に使うときは、logits, lossの算出だけautocastで囲む。
    ```python
    for x, y in dataloader:
        optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(logits, y)
        
        loss.backward()
        optimizer.step()
    ```
- p257. 勾配クリッピング：勾配が大きくなりすぎた場合、向きはそのままに大きさをクリップする。
    ```python
    grad_clip = 1.0
    loss.backward()
    torch.nn.util.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    ```
- p260. 事前学習の流れ
    1. train_data, val_dataは事前にtokenizeしておき、np.memmapを使って順次呼び出す。
    2. lr_schedulerから学習率を取得
    3. batch分だけデータを取得。
    4. torch.autocastを使ってlogits, lossを算出。
    5. loss.backward()
    6. clip_grad_norm_()
    7. optimizer.step()
    8. 固定iterationごとにモデルを保存。また、モデルを評価。
- p264. 評価：色々あるが、LLM-as-a-Judgeを使う場合、10件くらいのサンプルを生成し、それぞれで評価し、平均スコア+ｰ標準偏差で表すことが多い。
- p268. DPOの式変形：人間の好みをLLMに学習させたいと考えた時、選好データ（Preference Data）を使ってBradley-Terryモデルを仮定することで、y_w, y_lの報酬の差を求めれば学習できることがわかる。学習方法には色々あるが、DPOでは式変形により報酬計算の過程で登場した正則化項Zが消せることが示されている。これにより、報酬から方策を導けるし、方策から報酬を導ける。
- p274. DPOのLossを算出するときは、正解応答と失敗応答それぞれにおいてprompt+response+paddingで固定長のidsを作る。それをモデルに入力し、全体のlog_probabilityを算出する。lossとして必要なのはresponse箇所だけなので、そこだけを抽出するmaskを作り、正解とするtokenに対するlog_probを抽出して和を取る。これで正解応答に対するlogprobsと、失敗応答に対するlogprobsが算出できたので、DPOの損失を計算できる。


# 7章 WebBot Tokenizer

- p284. 高速化のためには計測が大事。`python -m cProfile -s cumulative sample.py > profile.txt`で最も時間がかかっている処理を特定し、Rust/C++で書き直すなどで高速化する。huggingfaceのtokenizersはRustで書かれているので早い。
- p285. 特殊トークンは自由に追加できる。Harmony形（gpt-ossで採用されている）では、<|channel|>, <|message|>などが使われている。
- p286. tokenizeの方法は他にも色々ある。SentencePieceというサブワード分割ライブラリは、2018年に開発され、LlamaやT5などで使われている。WordPieceはBERTで採用されたもので、BPEとはマージ方法が異なる。

