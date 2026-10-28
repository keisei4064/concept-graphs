# ローカル実行の結果と制約

環境は[ENVIRONMENT.md](ENVIRONMENT.md)、再現コマンドは[RUNBOOK.md](RUNBOOK.md)。
保存・参照整合性・可視化の合格と、推論の意味的正確性は別に判定する。GTによる正解率は未測定。

## 実行結果

| 実験 | 入力 | 保存物体 | 保存関係 | 状態 |
|---|---|---:|---:|---|
| M1 VLMなし | 先頭3枚 | 17 | 0 | 保存・再ロード・構造検査合格 |
| M1 VLMなし | 先頭100枚 | 25 | 0 | 同上。関連検出1980件 |
| M3 Qwen35B | 先頭3枚 | 17 | 3 | 関連検出61件、統合Caption17件 |
| M3 Qwen9B | 同じ先頭3枚 | 17 | 1 | 同上。16物体で複数観測を統合 |
| Qwen35B pilot | 全区間stride20・100枚 | 61 | 25 | 関連検出1931件、1146.5秒 |
| 全2000枚の検出・mapping | stride1 | 71 | — | 検出完了、関連検出37491件 |
| 全2000枚のScene Graph | stride1 | — | — | frame000099で停止、再開保留 |
| Qwen35B reference | 全区間stride10・200枚 | 73 | 39 | 問題付き完了、関連検出3805件、1803.5秒 |
| 公開main参考の後処理 | referenceの73物体 | 43 | 12 | MST候補21組、意味的方向誤りあり |

M0は専用環境のimport・GPU演算・PyTorch3DのCPU/GPU IoUに合格。
M1/M3は点群・特徴の有限性、UUID一意性、エッジ端点、保存・再ロードを検証した。
ネイティブRerun GUIのRGB・注釈・3D・BBox・関係表示も確認済み。過去のGUI確認は現在の稼働を意味しない。

M2はframe000000の同じ20検出付き画像で検証した。temperature0、max_tokens4096、context8192、thinkingなし。

| モデル | 関係数 | Caption数 | 関係／Caption／統合の秒数 |
|---|---:|---:|---|
| qwen3.5:9b | 2 | 20 | 1.114 / 9.759 / 0.623 |
| qwen3.5:35b | 6 | 20 | 1.654 / 6.677 / 0.595 |

いずれも100% GPUロードを確認。値は成功した試行のみで、warm cache・失敗・初回ロードの影響を除いた速度比較ではない。
9Bのdigestは`56671c2ab938`、35Bは`72aac7c28be8`。応答・完全digest・遅延は`outputs/local-only/m2/`に保持。
枕→ソファや植栽容器→机は支持できる一方、横並びの扉→扉、皿→遠方の机などの誤りがある。
単一物体のCaptionへ異なる家具が混ざる例もあり、検出・融合とVLMのどちらに原因があるかは未分離。

## 200枚referenceとCaption救済

成果物：`outputs/local-only/assets/data/Replica/room0/exps/room0_reference_graph_attempt1/`。
元RRD約3.2GiB、最終pkl.gz約27MiB。状態は`completed_with_vlm_issues`。

- 応答全体の失敗0件、関係7件・Caption項目1715件を除外。入力IDに対するCaption欠落1887件、空の統合Caption16件。
- 除外Captionの1712件は`caption`キー欠落等。確認例では`description`に説明文があり、内容の認識失敗とは区別する。
- 別exportで1712件を救済し、入力IDに対する欠落1887→175。最終マップでは1059観測を救済し、欠落2078→1019／3805観測。
- 元地図・キャッシュ・RRDは不変。再推論・統合Caption再生成・GUIへの反映は行っていない。
- 救済先：`outputs/local-only/reference_room0/caption_recovery/`。旧2000枚停止状態は`outputs/local-only/full_room0/state.json`。

100枚pilotは旧strict診断方針による実験。Caption100応答の採用は初回70・再試行30、関係は初回94・再試行6。
この形式検証の集計を、現在のupstream方針の成功率や意味的正解率として扱わない。

## 公開mainを参考にした後処理

完成済み73物体の保存観測から、画像Caption→物体要約→点群重複率→最小全域森→関係推定を別実装で実行。
3タスクともQwen35B。元の3関数を直接実行する薄いアダプター構成ではなく、プロンプト抽出と既存重複関数の再利用を含む。
固定参照ソースとURL・SHA256は`outputs/local-only/paper_scenegraph_room0/attempt1/source_refs/`。

| 段階 | 結果 |
|---|---|
| 観測選択 | 2観測未満21物体を対象外。confidence上位最大10観測、padding10、maskingなし |
| 画像Caption | 460選択観測から面積4900未満76件を除き384画像 |
| 要約 | 52物体分を生成し、invalid等9物体を除いて43物体 |
| 候補 | 重複25組→22連結成分の最小全域森21組→LLM21組 |
| 関係 | 採用12件（全てon）・該当なし9件、API／解析／Schema失敗0件 |

最終結果：`outputs/local-only/paper_scenegraph_room0/attempt1/full_main/`。`full/`はBBox軸順修正前の中間結果。
初回756.8秒、BBox再ロード後の軸順を修正して453キャッシュを再利用し4関係のみ追加推論した。
既存39エッジをonへ正規化した35有向関係との方向込み一致は6件。正解率・改善率を示す比較ではない。

植栽容器→台は画像と整合するが、椅子→枕・ソファ→枕は逆向き。
理由文が「枕が家具の上」と説明していてもenumが`a on b`の例を保持し、自動反転していない。
実例は`semantic_review.md`、根拠画像と全文は`evidence.md`、比較は`comparison.json`。

公開mainは点群近接率（距離0.025m、上三角0.01超）を直接の重みとして**最小化**する。
論文本文のBBox IoUとは異なり、強い重複を優先する最大化や重み反転も行っていない。
関係推定へ渡すのは`object_tag`とBBoxで、summary全文や画像ではない。
観測選択も公開mainのconfidence順であり、論文のノイズ除去後の寄与点数順ではない。

## ali-devからの変更と検証範囲

- ローカルQwen接続、JSON受け渡し、応答・画像参照・モデル設定の記録、ループバック制限を追加。
- 通常の`upstream`方針はAPI／解析失敗を記録し、不正項目を除外して継続。欠落Captionはnull補完し、重複IDはlast-wins。自動再生成0回。
- 独自の名前固定や全ID必須条件はstrict診断へ分離。`description`の別名救済は通常パーサには入れずオフラインexportに限定。
- フィルタ後の検出index、新規追加前の物体数、エッジの内部indexとUUID参照、VLM無効時の戻り値を修正。
- 注釈が画像外へ切れないよう配置を修正。入力画像のピクセルは変わる。
- 初期物体登録フレームの関係不採用、表示閾値2、消えた物体のエッジ除外は元の方針を保持。統合先へのエッジ引き継ぎは追加していない。
- 原子的保存、raw検出キャッシュ再利用、進捗監視を追加。失敗した融合は先頭から再計算し、途中の3D状態からは直接再開しない。
- 監査時に103対象を比較し主要16対象を公式ali-devとも照合。幾何・類似度・融合等11項目をAST比較。証拠は`outputs/local-only/source_audit_20261026/`と`alignment_validation/`。

これらは論文全体や数値結果の一致を意味しない。今回のmappingはvoxel1cm・association閾値1.2・cosineそのままの加算で、
論文の2.5cm・1.1・`cos/2+1/2`とは異なる。ローカルのmap同士のmergeは`text_sim = visual_sim`であり、独立したテキスト類似度判定ではない。
発表素材との対応・その他の差は[素材README](../発表/素材_20261027/README.md)を参照。

現行78テスト合格。後処理では384切り出しの画素一致、176元画像・元地図のハッシュ、457応答、NetworkXによるMST最小コストを検証。
21キャッシュから新規推論0での再開も確認。可視化は638701点・43BBoxを保持し、段階間のノード位置を固定する。
通信監査は推論が127.0.0.1のみ。GUI初回の外向き接続試行を検出した後、専用設定とbwrap隔離を追加し、隔離後は接続先なし。
既存環境・通常Ollamaサービス・元データは変更しない。監査・検証の生記録は`outputs/local-only/`に保持する。
