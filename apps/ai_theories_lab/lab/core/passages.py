"""合成の負荷の書き出しに使う英文(コーパスを持たないアプリのための、短い固定の文章)。

027 のノートブックは、Wikipedia の検証用の記事の先頭から書き出しを切り出した。このアプリは
コーパスを取得しないので、同じ役割(長さの違う書き出し、全要求に共通の system prompt)を、
ここの文章から作る。内容は一般的な百科事典風の英文で、モデルの性能とは関係しない。
"""

PASSAGE_TEXT = (
    "The history of the city began when a small group of traders settled beside the river and "
    "built a market where roads from the hills and the coast met. Over the following centuries "
    "the town grew into a port, and its merchants carried grain, wool, and timber to distant "
    "harbours. A stone bridge replaced the old ferry, and the surrounding villages sent farmers "
    "and craftsmen to sell their goods every week. Albert Einstein was born in a small town in "
    "the south of Germany, and the family moved several times before he finished school. He "
    "studied physics, worked in a patent office for several years, and published papers that "
    "changed how scientists think about space, time, and light. The film was released in the "
    "spring, and critics praised its photography, its music, and the quiet performance of the "
    "lead actor. It tells the story of a young teacher who travels to a remote island to open a "
    "school, and who slowly learns the customs of the people who live there. The tokenization "
    "of words into smaller pieces lets a language model read rare names and unexpected "
    "spellings without a vocabulary of every possible word. The river flows north for more than "
    "four hundred kilometres before it reaches the sea, passing through forests, farmland, and "
    "several large towns. In the early twentieth century a railway line followed the valley, "
    "and the station became the centre of a new neighbourhood with shops, a library, and a "
    "public garden. The museum holds a collection of maps, instruments, and letters that "
    "describe how the region was surveyed, and visitors can follow the work of the first "
    "cartographers through drawings made in the field. The team won its first national "
    "championship after a long season, and the final match was watched by a record crowd. "
    "During the winter the lake freezes, and local families skate across it to reach the "
    "island where an old lighthouse still stands. The orchestra performed the symphony for the "
    "first time in the capital, and the composer revised the final movement twice before it was "
    "published."
)
"""書き出しを切り出す英文。"""

SYSTEM_PROMPT_TEXT = (
    "The following article is written in the neutral style of an encyclopedia. It states facts "
    "plainly, avoids opinions, and cites the places and the dates that it mentions. Read the "
    "opening of the article and continue it in the same style. "
)
"""全要求に共通の接頭辞(system prompt を想定)の元の文章。"""
