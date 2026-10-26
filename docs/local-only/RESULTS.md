# 実験結果 (2026-10-25 JST)

## M0 / M1

M0: PASS (`b5d041d`)、M1: PASS (`dda5d68`)。
M1は原ali-devのYOLO-World/SAM-L/ViT-H-14 OpenCLIPと幾何・対応付け・融合を使用した。

|入力|保存物体|関連検出|外部通信|保存・再ロード・Rerun構造|
|---|---:|---:|---|---|
|room0先頭3フレーム、stride 1|17|61|なし|PASS|
|room0先頭100フレーム、stride 1|25|1980|なし|PASS|

全入力のRGB-D組、depth scale、intrinsics、c2w、投影往復は検査済み。
M1時点のRerun検証は記録ファイル・timeline/点群/box/IDの構造検査。追加GUI検証は末尾に記載。
元実装の物体分類・融合の精度を保証する検証ではない。

## M2: 独立した実画像テスト

同一入力: `outputs/local-only/m2/input-readable/annotated.jpg` (20番号付き検出)。
SHA256: `f4b5fbd24bc33191aafb5873d4f642b6e9a8caea1ca23bd2ae636bb8a5f6ae0d`。
Replica frame000000の元検出キャッシュから生成し、画面端で番号の先頭桁が切れる問題のみ修正。
temperature 0、max_tokens 4096、thinkingなし、context 8192、量子化は既存タグのデフォルトのまま。

|モデル|関係数|Caption数|関係 秒|Caption 秒|統合 秒|GPU|合否|
|---|---:|---:|---:|---:|---:|---|---|
|qwen3.5:9b|2|20|1.114|9.759|0.623|100% GPU|PASS|
|qwen3.5:35b|6|20|1.654|6.677|0.595|100% GPU|PASS|

9B digest: `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`。
35B digest: `72aac7c28be8ff9b81c2a270eb489c88e6d822a76ec0cb7bf34d965768848567`。

上記は検証に成功した再試行の実測値で、warm model/image cacheの影響を含む。
初回ロードや全失敗試行の時間を含む性能比較ではなく、速度・精度の優位性は主張しない。
9Bの全3タスクは11.65秒、35Bは9.10秒。
統合単体入力は各モデルのID 0の実Caption 1件。複数観測の統合はM3で検証する。

合格成果物:

- `outputs/local-only/m2/qwen9b-final/summary.json`
- `outputs/local-only/m2/qwen35b-readable/summary.json`
- 各`responses/`に全プロンプト・schema・生応答・JSON・遅延・モデルdigest・画像hash。
- 各`ollama-ps.txt`, `ps.json`, `nvidia-smi.txt`にGPUロード実測。
- 各`network-audit.json`と`m2/server/network-audit-m2.json`: 127.0.0.1のみ、外部宛てなし。
- 一時サーバーの`status.json`: `cloud.disabled=true`, `source=env`。

失敗は合格と区別して保持:

- `m2/qwen9b/`: legacy出力指示とschema指示が混在し、不正JSONを返した。約40.73秒。
  空グラフに変換せず例外化。ローカル用プロンプトの出力指示をJSONに統一した。
- `m2/qwen9b-schema/`: ID 8のCaptionをposterとして返したため拒否。
  元画像の端でID 28の先頭桁が切れることを確認し、注釈位置を画像内へ制限した。
  ID/nameの対応もschemaで固定し、同一の修正画像で9B→35Bを検証した。
- `m2/qwen35b-final/`: JSON本体は有効だがMarkdown fence付きで厳密json.loadsが失敗。約27.41秒。
  単一fenceの明示的正規化を追加し、JSON/ID/Relation検証を通した再試行が上表。

## 意味の目視点検 (自動スキーマ合格とは別)

この1枚の注釈画像を目視点検した。モデル全体の精度評価ではない。

- 35Bの `pillow 2 on top of sofa chair 0` と `potted plant 6 on top of coffee table 13` は
  画像から支持できる。
- 35Bの `closet door 8 on top of closet door 24` は左右に並ぶ扉と整合しない。
- 35Bの `plate 16 on top of coffee table 32` は部屋の反対側の机へ結び付き、誤った対応。
- 9Bの `poster 28 under window 9` は直接の物理的関係として画像から支持できない。
  `lamp 7 on top of coffee table 13` も支持面を確定できず、未観測関係の可能性がある。

この意味的な誤りを隠すためにエッジを手動で作成・置換していない。
M3でも参照整合性と可視化の合否を、物理的正しさから分けて報告する。

## M3: 保存済み検出を利用したEnd-to-end

両モデルともReplicaの同じ先頭3フレーム (stride 1) を使用した。
M1のraw検出/マスク/特徴は不変で、3D融合とVLM統合は元の処理順で実行した。

|モデル|物体|関連検出|保存エッジ|統合Caption|複数観測を統合した物体|保存・再ロード・可視化|
|---|---:|---:|---:|---:|---:|---|
|qwen3.5:35b|17|61|3|17|16|PASS|
|qwen3.5:9b|17|61|1|17|16|PASS|

35Bは`.../room0/exps/m3_qwen35b_complete/`、9Bは`.../room0/exps/m3_qwen9b/`に
元形式pkl.gz、ノード/エッジJSON、設定、対応付けログ、frame metadataを保存。
`outputs/local-only/m3/summary.json`に集計、各モデルの`map-validation.json`に整合性検査。
両モデルで画像上の検出ID→3D物体ID→有効なエッジ端点を確認し、Caption名を元class_idと照合した。

Rerunの元観測閾値2では単一観測のエッジが非表示になるため、閾値1で保存地図を再記録した。
各`m3/<run>/scene_graph_full.rrd`をSDK 0.15.1で読み、点群/box/uuid/frame/関係線を確認。
同じ保存物から作った`scene_graph.png`も目視した。この時点ではGUI viewerは未確認だったが、追加検証で表示を確認した。

35B最終グラフの関係は `potted plant on top of coffee table` (観測1)、
`lamp under light switch` (観測2)、`end table under potted plant` (観測1)。
9Bは `lamp on top of coffee table` (観測1)。これらのうちlamp/遠方のlight switchの関係や、
支持面の見えないlampと机の関係は物理的関係として支持できない。意味的誤りは残る。
可視化できること・参照整合性と、意味的正確性を同じ合格判定にしていない。

有効エッジの一部は元のfilter/mergeで消え、最終数はフレーム別のVLM回答数とは一致しない。
元設計の採否・融合を維持し、都合のよい回答を選んだりエッジを補填したりしていない。

最終ネットワーク監査: M1/M3 VLM無効と描画/再記録は接続先ゼロ、M2/M3推論と
一時Ollamaサーバーは`127.0.0.1`のみ。APIキー不要、重みの通常実行時ダウンロードなし。
停止endpointを指定したM1再検証も17物体/61関連検出でPASS。
保護ファイル全件のSHA256不変と、元Ollamaサービスの稼働継続を確認。

最終テスト32件合格。失敗したJSON/ID/schema echo/対応付けとログの問題、
キャッシュの制約、再現コマンドは [RUNBOOK.md](RUNBOOK.md) に記載。

## 追加のGUI表示確認

2026-10-25、ユーザーの希望により35B保存地図をRerun 0.15.1のネイティブGUIで開き、
実際の描画を`outputs/local-only/gui/rerun35b.png`で確認した。
通常起動のRerun Viewerは開いたままで、ユーザーが回転・拡大やentity表示切り替えを操作できる。
外部通信なし、既存設定への変更なし。自動スクリーンショット終了時のpanicと通常起動の区別はRUNBOOKに記録した。

## 全区間100枚試験でのQwen 35Bの実用性点検

`room0_pilot_graph_attempt2`はstride 20で全区間の100枚を処理し、61物体・25関係を保存した。
これは構造・保存・再ロード・表示の合格であり、意味的正確性の合格ではない。
採用された応答キャッシュの試行番号と遅延を集計した。

|タスク|採用応答|試行1で採用|試行2で採用|試行3で採用|採用応答の遅延中央値|
|---|---:|---:|---:|---:|---:|
|Caption|100|70|30|0|7.21秒|
|関係|100|94|5|1|2.10秒|
|Caption統合|56|56|0|0|0.59秒|

ID欠落・対象外ID・`caption`の代わりに`description`を返すなどの違反があり、
検証とローカル再生成が必要だった。初回停止を含む2回のpipeline実行を通算して、
失敗応答はCaption 33件・関係7件を保存している。
試行番号は形式・ID検証の指標であり正解率ではない。遅延は失敗試行・モデルロードを含まない。
61物体中5物体は統合対象の観測Captionがなく、理由を記録してnullのままとした。

画像説明には金属製ケトル・黒い取っ手・木製cabinetなどの具体的な記述がある。
一方、統合後のend table #13は「二種類の家具」、pillow #491は「複数の異なる枕」を
列挙しており、単一物体の説明としての品質には課題がある。元の検出・対応付け・融合で
異なる観測が混ざる可能性もあるため、Qwenだけの誤りとは断定しない。
M2の枕→ソファ、植木→机は支持できるが、横並びの扉→扉、皿→離れた机の誤りも残る。

現状は画像説明・関係候補の生成には使えるが、無検査で正しいScene Graphとして扱う根拠はない。
9Bとの比較は同一の1画像と先頭3枚のみ。GPT系との同条件比較・GTによる意味的正解率は未測定。
詳細集計は`outputs/local-only/full_room0/qwen-quality-review.json`。

## 原処理に合わせた200枚実走 (2026-10-26)

`room0_reference_graph_attempt1`、全区間stride10・200枚、Qwen3.5:35b。
10:06:12 JST開始、10:36:47 JSTに保存物検証まで完了。mapping1803.5秒。
73物体・39関係・3805対応検出、全200枚のraw検出キャッシュを再利用。
地図再ロード・参照整合性・Rerun構造・通信監査が合格。接続先は127.0.0.1のみ。
成果物は`outputs/local-only/assets/data/Replica/room0/exps/room0_reference_graph_attempt1/`。
元Rerun約3.2 GiB、最終pkl.gz約27 MiB。

完了状態は`completed_with_vlm_issues`であり、全件正常・意味的正確性の合格ではない。
VLMの応答全体の失敗は0だが、関係7件・Caption項目1715件を除外し、入力IDに対するCaption欠落1887件、
空の統合Caption16件を記録。保存地図ではCaption欠落が2078/3805観測。
除外Captionの1712件はcaptionフィールドがない/文字列でないという理由。確認したサンプルは
`description`キーに説明文が入っているため、記述の意味的な失敗とは区別する必要がある。
元の説明文・応答・プロンプトはVLMキャッシュに保持。後日の別ファイルへの救済結果は下記。
残り3件は部分解析時のCaption項目エラー。


## 保存Captionのオフライン救済

`recover_saved_captions.py`で、保存応答の`description`を別ファイルの`caption`として取り出した。
再推論・元地図の書換え・統合Captionの再生成は行っていない。

- 1712件の説明文を正規化し、VLM入力IDに対する欠落は1887→175。
- 最終マップでは1059観測を救済し、3805観測中の欠落は2078→1019。
- 元地図と応答200件の計201ファイルはSHA256不変。通信監査も合格。
- 出力: `outputs/local-only/reference_room0/caption_recovery/`。

これは形式・項目の救済件数であり、意味的正確性の評価ではない。
元RRDには反映していない。実行法は[RUNBOOK.md](RUNBOOK.md)を参照。
