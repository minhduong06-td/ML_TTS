# LJSpeech dataset audit

The workspace is named LJSpeech. The local files alone do not establish the source download or license; verify the provider and its terms before redistribution. Original dataset ZIP: [Drive backup](https://drive.google.com/file/d/1TOZFnzbs0DzUFwl35LkojI66px9ar5Tq/view?usp=drive_link). The backup contents have not been inspected.

The official English baseline uses `data/en_ljspeech/metadata.csv`: UTF-8 physical lines with three `|`-separated fields: WAV ID, original transcription, normalized transcription. The formatter reads the third field when nonempty and resolves the ID as `wavs/<ID>.wav`; it does not interpret CSV quoting. Example from the actual file:

`LJ001-0002|in being comparatively modern.,|in being comparatively modern.,`

Records require exactly three columns, a safe unique ID, nonempty text and a matching WAV. The current audit found 13,100 valid records, no duplicates or missing WAVs, and 13,100 readable headers, all mono at 22,050 Hz. Headers do not establish decoded audio quality. The machine-generated measurements and duration distribution are in `dataset_statistics.json`.
