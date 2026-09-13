## CLI Usage

Run drift detection against an input dataset:

```bash
drift-detector \
	--data "data/Online Retail.xlsx" \
	--split-date 2011-07-01
```

View all available options:

```bash
drift-detector --help
```

Write reports to a custom directory:

```bash
drift-detector \
	--data "data/Online Retail.xlsx" \
	--split-date 2011-07-01 \
	--output-dir reports/custom
```
