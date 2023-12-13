### Installation

#### Prerequisite

> Make sure you have at least python 3.9 installed.

> We use [pdm](https://github.com/pdm-project/pdm) as package manager for this project.

For Linux, MacOS, Windows (WSL)

```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

For Windows (PowerShell)

> Note: if you installed python through Microsoft Store, use "py" instead of "python" in the following command.

```bash
(Invoke-WebRequest -Uri https://pdm-project.org/install-pdm.py -UseBasicParsing).Content | python -
```

Other ways to install it (e.g. using pipx): https://pdm-project.org/latest/#installation

#### Installing the packages

```bash
pdm install --no-self
```

#### Installing resources (datasets and images)

If you have the _make_ GNU utility, you can perform an automatic installation of the resources.

```bash
make resources
```

Alternatively, you can install the resources manually.

> Create a `data` folder at the root of the project and put in there the datasets. Then run the precomputing script that will split the datasets and compress them.

```bash
python precompute_datasets.py
```

> Then create a `assets` folder and add the `img_celeba` folder in it.

#### Running the project

Using the following command, it will print the command you need to type to activate the environment.

```bash
pdm venv activate
```

Then type the command given to you.

Here is an example.
![example-activating-venv](./example_venv_activate.png)

Finally you can run the project.

```bash
python main.py
```

#### Troubleshooting

> Could not build wheels...

```bash
pip install --upgrade pip setuptools wheel
```
