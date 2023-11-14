### Installation

> We use Poetry as package manager for this project

```bash
$ pip install poetry
$ poetry install
$ poetry shell
```

> After installing Poetry and running the virtual environment, create a `data` folder at the root of the project and put in there the datasets. Then run the precomputing script that will split the datasets and compress them.

```bash
$ python precompute_datasets.py
```

> Then add the `img_celeba` folder in the assets `folder`.
> Finally, you can run the project

```bash
$ python main.py
```
