# 発表素材

公開コードを参考にしたローカル実行・説明用の素材。**原論文・公式実装そのままの出力ではない**。
基準はAの椅子（Map #1070、frame001070）。画像・点群・Captionを同じ物体に揃えて使う。
画像・PDF・抽出文・検証JSONはGit対象外で、以下のリンクはローカル素材がある場合に参照できる。

## スライドに使う部品

| 用途 | 素材 |
|---|---|
| p9・p12：元RGB | [rgb.png](A_chair/rgb.png) |
| p9・p12：椅子のSAMマスク | [二値](A_chair/sam_01_binary.png)／[黒背景RGB](A_chair/sam_01_rgb_black.png)／[透過RGB](A_chair/sam_01_rgba.png)／[元画像に色重ね](A_chair/sam_01_overlay.png) |
| p12：RGB×maskを説明する3枚 | [RGB切り出し](A_chair/sam_01_rgb_crop.png) → [同じ範囲の二値マスク](A_chair/sam_01_binary_crop.png) → [乗算結果](A_chair/sam_01_masked_crop_black.png) |
| p12：実際にCLIPへ送った画像 | [選択した画像](A_chair/clip_same_01.png)／[同じMap物体の別観測](A_chair/clip_same_02.png) |
| p15・16：逆投影用のマスク適用画像 | [物体周辺の透過RGB](A_chair/masked_rgb_crop.png)／[p15用に傾けた版](A_chair/masked_rgb_crop_p15.png)／[全体画像上の使用領域](A_chair/mask_01_overlay.png) |
| p15・16・17：ノイズ除去前の点群 | [projection_before_front.png](A_chair/projection_before_front.png) |
| p15・17・19：ノイズ除去後の点群 | [projection_after_front.png](A_chair/projection_after_front.png)／[RGB色の点群](A_chair/projection_rgb_front.png) |
| p19右上・p20左：融合前 | [fusion_before_front.png](A_chair/fusion_before_front.png)。既存が青、新規が橙、両方BBox付き |
| p20右：融合後 | [fusion_after_front.png](A_chair/fusion_after_front.png) |
| p23：複数観測の画像 | [候補一覧](A_chair/caption_choices.jpg)／[観測1](A_chair/caption_01_crop.png)／[観測2](A_chair/caption_02_crop.png)／[観測10](A_chair/caption_10_crop.png) |
| p23：対応するCaption | [1](A_chair/caption_01_card.png)／[2](A_chair/caption_02_card.png)／[10](A_chair/caption_10_card.png)。全文は同名の`_full.txt` |
| p25：この椅子と青い枕の関係推定入力 | [入力JSON](relation_text/41_44_input.json)／[実際のプロンプト全文](relation_text/41_44_prompt_full.txt)／[元のモデル応答](relation_text/41_44_response.json) |
| p23右端・p28：椅子の要約 | [JSON](A_chair/summary.json)／[横長の貼り付け用画像](A_chair/summary_inline_card.png) |

p12の別物体画像は[clip_different.png](A_chair/clip_different.png)。p9の複数物体マスクは[masks_choices.jpg](A_chair/masks_choices.jpg)から選ぶ。
`sam_*`は元SAM出力、`mask_*`／`masked_rgb*`は包含物体の差し引き後の逆投影用マスク。
`clip_*`は背景を含む実CLIP入力。RGB×maskの説明用画像とは異なる。

点群は`front`／`oblique`、透過／末尾`_white`の白背景版を用意。前後比較では視点名を揃える。
テキストカードの編集には同名のSVG、Caption全文には`_full.txt`を使う。
選択椅子の逆投影は6,444点→voxel化5,524点→DBSCAN後5,514点。青い枕は別検出なので穴は補完しない。
融合例は過去3観測（frame630・640・700）＋frame1070の部分再現で、当時の全マップではない。

比較候補は[Bのソファ](B_sofa/preview.jpg)（Map #1282、除去140点）と[Cの植栽容器](C_planter/preview.jpg)（Map #6、除去0点、Caption9件）。
重なった赤枠は複数候補の指定であり、候補数の固定ではない。追加のカメラ合成図とNode Captioning一枚図は推奨部品に含めない。

p25の別候補：[植栽容器と台](relation_text/2_7_prompt_full.txt)／[ソファと枕](relation_text/19_29_prompt_full.txt)。
各`*_response.json`が実応答。ソファのMap #134はB候補のMap #1282とは別物体。

## 再生成

既存の`conceptgraph-local`環境とローカル実験データが必要。新規のモデル推論は行わない。
原稿PDFは`docs/発表/`に1件置くか、`CG_SLIDES_SOURCE_PDF`にリポジトリ内の対象パスを指定する。

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  conda run --no-capture-output -n conceptgraph-local \
  python docs/発表/素材_20261027/build_assets.py
```

基本素材A〜C・関係テキスト2組・候補一覧を更新する。後から追加した`41_44`、p15用アフィン変形、
Node Captioning一枚図、追加検証JSONは再生成しない。`build_p15.py`は不採用の追加合成図用。
[manifest.json](manifest.json)に入力と素材のハッシュ、各`provenance.json`に点群・融合の条件、
[validation.json](validation.json)と[selected_chair.json](selected_chair.json)に検証結果を保存している。

## ページ別の判定

| ページ・素材 | 確認できたこと | 原論文／公式そのままとは言えない点 |
|---|---|---|
| p9：RGB・`sam_*` | Replica room0の保存画像と、保存されたSAMマスクに画素一致 | 今回はYOLO-WorldのBBoxをSAMへ入力。SAM Automatic Mask Generationの出力例とは呼べない。SAM自体のクラス非依存性と、今回のBBox候補の作り方は分ける |
| p12：`clip_*` | 保存されたCLIP入力画像と一致 | 実際はpadding 20のBBox切り出しで背景も含む。`sam_*masked_crop*`はRGB×maskの説明用画像で、この実験のCLIP実入力ではない |
| p15〜17：点群 | ローカルmappingの既存関数と保存RGB-D・マスク・姿勢から再計算。選択した椅子の6,444点を再投影して同じマスク内・同じDepthであることを確認 | 当時の中間点群ファイルの抽出ではない。灰色は最大クラスタ外として除去された点で、真のノイズと確認された点ではない |
| p15：ひし形画像・追加合成図 | ひし形PNGは元画像の2Dアフィン変形。元画像は不変 | 実カメラの投影幾何から画像平面を描いたものではない。別途作ったカメラアイコン・座標軸の配置図も説明用で、今回の使用対象から除外済み |
| p19〜20：融合前後 | 元の`merge_obj2_into_obj1`を使用。選択椅子の過去3観測にframe001070を融合 | 当時の全履歴マップのスナップショットではない。図生成時にはAssociationの候補検索・勝者選択を実行していない |
| p21：既存物体同士のMerge説明 | 保存した公開mainには重複・画像特徴・テキスト特徴の3条件がある | 参照中のローカルmappingコードでは`text_sim = visual_sim`。独立したテキスト類似度判定を実行したとは説明できない |
| p23：画像・Caption・要約 | 同じMap物体の保存観測とQwen応答。10件の全文が要約要求に入ったことをキャッシュで確認。選択数・切り出し等は公開mainに合わせている | LLaVA／GPT-4ではなくQwen。JSON Schema制約も追加。公開mainと今回のビュー選択はconfidence順だが、論文の選択基準はノイズ除去後の寄与点数 |
| p25：MST候補図・関係推定 | 保存した公開mainを基準に点群近接率・上三角・直接重みの最小全域森を再実装し、独立実装で最小コストを検証 | 論文本文は3D BBox IoU。候補生成は元関数の直接実行ではなく別実装。モデルもQwenで、完全な挙動同値性を網羅的に証明したものではない |
| p25：プロンプト・出力 | `*_prompt_full.txt`と`*_response.json`は実際の保存入出力 | 会話で示した短いプロンプトは表示用の要約。採用されたエッジが意味的に正しいとは限らない |
| p28：JSON例 | `summary.json`は実際のQwen要約。`summary_inline_card.png`は全フィールド・値を保持した表示 | 会話の`Beige chair.`などは手で短縮した文。ID=1/2への置換や`source/target/relation`の3行例は独自の模式表現。枕→椅子の向きへ直した例は実出力ではない |

## 特に説明を訂正・補足する箇所

1. **「best 10」の基準は論文と公開mainで違う。** 論文のII-A脚注4・付録A2は寄与点数。公開mainの`extract_node_captions`は`conf`を降順に並べる。今回の素材は後者。
2. **mappingの条件も論文実験の値ではない。** 論文II-Dのvoxel 2.5 cm・association閾値1.1に対し、今回の設定はvoxel 1 cm・閾値1.2。今回のコードは画像特徴のcosineをそのまま加算し、論文に記載された`cos/2 + 1/2`とは同じ式ではない。
3. **本文と公開mainの関係推定入力にも粒度差がある。** 論文はcaptionと3D位置と説明するが、保存した公開mainは`object_tag`とBBox中心・寸法を入力する。今回もsummary全文や画像は関係推定へ渡していない。
4. **p28の3行JSONは公式フォーマットではない。** 公開mainの`generate_scenegraph_json`は物体ごとに`id`・BBox・`possible_tags`・`object_tag`・`caption`を出力し、要約の`summary`を`caption`へ移す。会話で提案した2物体＋1関係は、構造を説明するための独自の省略例。
5. **椅子と枕の実出力は逆向き。** `41_44_response.json`は`a on b`（椅子→枕）。理由文は枕が椅子の上と説明している。模式例の枕→椅子を「実験結果」として載せない。

## 発表に添える短い注記

- 素材全般：`可視化例はローカル実行によるもの（YOLO-World + SAM、Qwen使用）。`
- 点群／融合：`保存観測と既存処理から再現した例。`
- p23：`公開mainに合わせconfidence上位を選択。論文は寄与点数で選択。`
- p25：`候補生成は公開mainの点群近接率に基づく。論文本文はBBox IoU。`
- 短縮JSON：`説明用の模式例（フィールド・文・IDを省略／整形）。`

「論文で提案された構成に沿う」と「論文の実験条件を再現した」は区別する。
また、テスト合格・画像ハッシュ一致・再投影一致は、元論文との方式全体の一致や推論の正解率を保証しない。

## 確認元

- [原論文PDF](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf)：II-A〜D、付録A2・A3、Listing 1・2。
- 保存した公開main：[Node Captioning／MST／JSON出力](../../../outputs/local-only/paper_scenegraph_room0/attempt1/source_refs/official_main_scenegraph.py)、[重複・Merge処理](../../../outputs/local-only/paper_scenegraph_room0/attempt1/source_refs/official_main_slam_utils.py)。取得元とSHA256は同じ場所の`sources.json`。
- 今回のmapping：[rerun_realtime_mapping.py](../../../conceptgraph/slam/rerun_realtime_mapping.py)、[model_utils.py](../../../conceptgraph/utils/model_utils.py)、[mapping.py](../../../conceptgraph/slam/mapping.py)、[utils.py](../../../conceptgraph/slam/utils.py)。
- [実際の実験設定](../../../outputs/local-only/assets/data/Replica/room0/exps/room0_reference_graph_attempt1/config_params.json)。
- [別実装した後処理](../../../conceptgraph/scenegraph/local_paper_scenegraph.py)。
- 素材の検証：[validation.json](validation.json)、[selected_chair.json](selected_chair.json)、[Node Captioningの出典](A_chair/node_captioning_p23_sources.json)、[融合図の出典](A_chair/provenance.json)。
