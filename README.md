### Installation

#### Prerequisite

> We use Poetry as package manager for this project

For Linux, MacOS, Windows (WSL)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

For Windows (PowerShell)

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Other ways to install it: https://python-poetry.org/docs/#installing-with-the-official-installer

#### Automatic installation

Make sure you have the _make_ GNU utility and _python_ >= 3.9 installed.

```bash
make install
make venv
make datasets
```

```bash
make venv
make run
```

#### Manual installation

Alternatively, if the above command does not work, you can install and run the project manually

```bash
poetry install
poetry shell
pip install "dash[diskcache]"
pip install "dash[celery]"
```

> After installing Poetry and running the virtual environment, create a `data` folder at the root of the project and put in there the datasets. Then run the precomputing script that will split the datasets and compress them.

```bash
python precompute_datasets.py
```

> Then add the `img_celeba` folder in the assets `folder`.
> Finally, you can run the project

```bash
python main.py
```
