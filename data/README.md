# Data

The repository does not redistribute PrimeVul source-code records. `scripts/fetch_primevul.py` downloads the three exact upstream files used in the study and verifies their byte counts and SHA-256 hashes before any benchmark is built.

Pinned Hugging Face dataset revision:

```text
4fd7158322872d711e90f091dbd8673ef32cb1be
```

| File | Bytes | SHA-256 |
|---|---:|---|
| `primevul_train.jsonl` | 470,918,175 | `5cadee7f3383124cc26ef6e67c8dd76c321e5461c94bef399a2f03a28767e6f5` |
| `primevul_valid.jsonl` | 63,564,497 | `7f0b4d05b063c20d5cf57478128538b8c0c36354e8def7f2d4febe6773b9dc7c` |
| `primevul_test.jsonl` | 66,064,689 | `c9dcab5ee897da36eccf0dbb3b510de2dab66a7dec681af29a7a28c8a53331c4` |

Build the data with:

```bash
python scripts/reproduce.py fetch-data
python scripts/reproduce.py build-data
```

This creates:

```text
data/raw/primevul_main/
data/processed/cwe_119_imbalance/
data/processed/cwe_119_noise_ratio_preserved/
```

The active generated data occupy about 3.4 GiB in addition to the 573 MiB raw download. `data/manifests/data_integrity_manifest.csv` records 638 canonical raw and generated files. JSONL files are checked byte-for-byte. Generated metadata JSON is checked after excluding its timestamp field.

The split and benchmark summaries in `data/manifests/` are committed because they contain counts and hashes rather than source-code bodies.

PrimeVul is fetched from its upstream distribution. Its records contain code originating in multiple software projects, so downstream users must follow the applicable upstream dataset and source-project terms.
