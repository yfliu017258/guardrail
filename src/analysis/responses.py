"""Turn a model run into response-direction guardrail data (``prompt,expected,response``).

    python -m src.analysis.responses results/<model run> -o processed_data/responses/<model>.csv
    python -m src.evaluation --config src/config/evaluation/guardrail_service.yaml --input-type response \
        --data processed_data/responses/<model>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..data_schema import Sample, write_csv
from .loader import Run, responses_from_model_run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args(argv)
    run = Run(Path(args.run_dir))
    if run.task != "model_safety":
        raise SystemExit(f"{args.run_dir} is a {run.task!r} run; a model_safety run is needed")
    df = responses_from_model_run(run)
    n = write_csv([Sample(prompt=r.prompt, expected=r.expected, response=r.response) for r in df.itertuples()], args.output)
    print(f"wrote {n} rows ({(df['expected'] == 'block').sum()} block) -> {args.output}")


if __name__ == "__main__":
    main()
