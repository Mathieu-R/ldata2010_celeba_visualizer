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

Alternatively, if the above command does not work, you can install and run the project manually.  
At the root of the project, run the following commands.

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

> Then create a `assets` folder and add the `img_celeba` folder in it.
> Finally, you can run the project

```bash
python main.py
```

#### Conda

Poetry does not work really well with conda. If you have conda installed, you can run the following

```bash
conda create -n venv
```

```bash
venv\Scripts\activate.bat # windows
venv\Scripts\Activate.ps1 # powershell
```

```
pip install -r requirements.txt
python3 main.py
```
