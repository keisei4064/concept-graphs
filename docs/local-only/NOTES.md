# 作業メモ・継続する判断事項

更新日: 2026-10-26

実行手順は[RUNBOOK.md](RUNBOOK.md)、結果は[RESULTS.md](RESULTS.md)、
原実装との差は[BEHAVIOR_AUDIT.md](BEHAVIOR_AUDIT.md)に集約する。
ここには次の作業に必要な指示と状態だけを残す。

## 現在の依頼範囲

- ユーザーは未コミット変更の整理を依頼。必要なリファクタリングと不要・重複メモの整理を含む。
- Gitの読み取りと変更整理を再開。コミットメッセージは日本語で提案する。
  現段階ではコミット作成・pushは行っていない。
- 論文方式は[PAPER_SCENEGRAPH_PLAN.md](PAPER_SCENEGRAPH_PLAN.md)を作成済み。
  実装・追加推論・長時間実験は未着手。整理作業に便乗して開始しない。

## 実験の状態

| 実験 | 状態 |
|---|---|
| M0〜M3 | 完了。詳細はRESULTSとRUNBOOK |
| 全2000枚・検出 | 完了、71物体・37491対応検出 |
| 全2000枚・Scene Graph | frame000099で停止。見積もりを聞いてから再開判断する指示が継続 |
| stride 10・200枚・Scene Graph | 保存・検証まで完了。73物体・39関係・3805対応検出、問題付き完了 |
| 保存Caption救済 | 再推論なしで別ファイルへ出力済み。元地図・元RRDは変更していない |

2000枚実験の状態は`outputs/local-only/full_room0/state.json`、
200枚実験は`outputs/local-only/reference_room0/state.json`。
200枚実験の完了は、旧2000枚ジョブの再開許可や完了を意味しない。
一時Ollamaは実験終了時に停止済み。既存サービスや他のプロセスは変更・停止しない。

## 保存物と解釈

現在の比較用マップ・元RRD・物体/関係JSON:
`outputs/local-only/assets/data/Replica/room0/exps/room0_reference_graph_attempt1/`

- 元RRDは約3.2 GiB。RGB・関係付き注釈画像・3D地図の時系列を含む。
- VLM元応答・入力画像参照・プロンプトは`outputs/local-only/full_room0/vlm_cache/`。
- 救済結果は`outputs/local-only/reference_room0/caption_recovery/`。
  `captions.md`が物体別一覧、`objects.json`が観測と出典、`summary.json`が件数と元ファイルハッシュ。
- `description`の説明文を取り出し、1059観測を救済。保存マップ内のCaption欠落は2078→1019。
  既存の統合Captionは再生成していない。GUIにも反映していない。
- 形式違反・ID欠落・空の統合文の件数は、意味的な認識失敗率や正解率ではない。
  既存のSchema要求だけでは返却キーが揃わなかった事実を踏まえ、実応答を検証する。

## 可視化・保存容量についての指示

- 元実装のRGB＋注釈付きRGB＋3Dの表示を優先する。画像内の検出IDと融合後のmap IDを区別する。
- 最終マップだけの静的表示を使う場合、RGBや時系列を省いていることを明記する。
- 現在の元RRDにはCaption本文が記録されておらず、説明文パネルはない。
- 200枚のGUIを開いた記録は`outputs/local-only/reference_room0/view/`。
  起動方法・blueprintはRUNBOOKを参照。過去の起動記録を現在の稼働確認として扱わない。
- ユーザーは確認後に大きな保存物の削除を検討しているが、削除依頼はまだない。
  RRDを消すと同期再生履歴が失われる。地図・画像・キャッシュ・共有モデルも勝手に削除しない。

## 論文方式についての合意

- 保存済み73物体マップを再利用し、物体観測の切り出しからCaption・要約・MST・関係推定を行う。
- 3タスクともQwenを使用。既存のフレーム全体Captionの流用やmapping再実行はしない。
- 元の選択・除外・MSTの方針を維持。互換性修正・ID修正と方式の変更を区別する。
- `ali-dev`リアルタイム版、論文方式、ローカル変更を「公式実装」の一語で混同しない。

## 対応上の注意

過去にGit操作不要の指示後のコミットと、RGB付き元記録より先に簡略可視化を提示する誤りがあった。
今後は現在の依頼範囲を優先し、元機能・実測済み事項・予定の機能を分けて説明する。
