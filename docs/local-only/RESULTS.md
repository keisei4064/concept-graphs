# 実験結果 (2026-10-25 JST)

## M0 / M1

M0: PASS (`b5d041d`)、M1: PASS (`dda5d68`)。
M1は原ali-devのYOLO-World/SAM-L/ViT-H-14 OpenCLIPと幾何・対応付け・融合を使用した。

|入力|保存物体|関連検出|外部通信|保存・再ロード・Rerun構造|
|---|---:|---:|---|---|
|room0先頭3フレーム、stride 1|17|61|なし|PASS|
|room0先頭100フレーム、stride 1|25|1980|なし|PASS|

全入力のRGB-D組、depth scale、intrinsics、c2w、投影往復は検査済み。
Rerunは記録ファイル・timeline/点群/box/IDの構造検査であり、GUI目視は未実施。
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
