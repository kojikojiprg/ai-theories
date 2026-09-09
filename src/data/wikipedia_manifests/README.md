# Wikipedia マニフェスト(Wikipedia Manifests)

`theories/`・`scripts/`から Wikipedia の記事を固定集合として取得するための、
記事タイトルとリビジョン ID(revision ID)の対応を記した JSON ファイルを配置する
ディレクトリ。タイトルとリビジョン ID の両方を固定することで、取得時点によらず
同一の入力が得られる(`src/data/text.py`の`load_wikipedia_corpus()`を参照)。

## ファイル一覧

| ファイル | 言語 | 記事数 | 用途 |
|---|---|---|---|
| `en_006_pretraining.json` | 英語(en) | 356 | 006(小型 GPT の事前学習)の英語条件用 |
| `en_009_scaling.json` | 英語(en) | 9826 | 009(スケーリング則)の学習グリッド用。`en_006_pretraining.json`の 356 記事を部分集合として含む(006・008 で取得済みのキャッシュを再利用できる) |
| `ja_005_tokenizer_legacy.json` | 日本語(ja) | 44 | 005(トークナイザ)の日本語ドメイン用 |
| `ja_006_pretraining.json` | 日本語(ja) | 80 | 006(小型 GPT の事前学習)の日本語条件用 |

## `en_009_scaling.json` の構成

- 006 で使った 356 記事(`en_006_pretraining.json`と同一)。
- Wikipedia の Special:LongPages(長大記事一覧、API `list=querypage&qppage=Longpages`)
  から、長い記事の順に選定した記事(009 第 2 ラウンドまでで 644 記事、
  009 第 3 ラウンド修正で追加 8827 記事、計 9471 記事)。

009 第 3 ラウンド修正での拡張は、前提条件 P1(IsoFLOP プロファイルの最小値が掃引範囲の
内点にあること)を満たすために必要な計算量予算を確保する目的で行った(詳細は
`theories/02_pretraining/009_scaling_laws.ipynb`の 5.3.1 節を参照)。既存記事の
タイトル・リビジョン ID は一切変更していない(追記のみ)。

## 取得不能なエントリへの対処方針(リビジョン削除の再発時)

Wikipedia のリビジョンは、記事の削除・統合・改名や版指定削除(revision deletion)に
より、後から取得できなくなることがある。`en_009_scaling.json`では、009 第 2 ラウンドの
本番実行時点で 1 記事(`List of foreign footballers in top leagues of former
Yugoslavia`、リビジョン ID `1370486518`)が`nosuchrevid`(指定したリビジョン ID が
存在しない)エラーで取得できなくなっていることが判明し、009 第 3 ラウンド修正で
マニフェストから削除した。`nosuchrevid`は再試行では解決しない永続的な失敗であり、
マニフェストが「記事とリビジョン ID を固定して再現性を担保する」ために存在する以上、
取得不能なエントリを残し続けることはマニフェストの目的に反する。

将来同種の事態(取得時にスキップが発生する)が起きた場合の対処手順:

1. `scripts/promote_canonical_corpora.ipynb`の取得ループ(またはコーパスを直接取得する
   各ノートブックの該当セル)を実行し、`skipped_articles`(取得できなかった記事とその
   理由)を確認する。理由が`nosuchrevid`など永続的な失敗であることを確認する
   (一時的なネットワークエラー・429 は`_fetch_wikipedia_revision_plaintext`が
   自動的に再試行するため、ここに残るのは再試行しても解決しなかったものだけである)。
2. 当該エントリを対象マニフェストの JSON から削除する。**既存の他のエントリは変更しない
   (追記・削除のみを行い、既存エントリのタイトル・リビジョン ID は変更しない)。**
   取得不能なエントリの削除は、そのエントリが一度も本文を返していない限り、コーパスの
   内容に対して無操作である(取得済みコンテンツの比較可能性を損なわない)。
3. 必要であれば、Special:LongPages から重複しない代替記事を選定し、現在のリビジョン ID
   を固定して追記する(`action=query&generator=querypage&gqppage=Longpages&
   prop=revisions&rvprop=ids`で、記事のランキング・現在のリビジョン ID を同時に取得
   できる)。
4. マニフェストを更新したコミットのメッセージに、削除したエントリ・理由
   (`nosuchrevid`など)・コーパス内容への影響(無操作であること)を明記する。
