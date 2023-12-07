POETRY := $(shell command -v poetry 2> /dev/null)
PIP := $(shell command -v pip 2> /dev/null)
UNZIP := $(shell command -v unzip 2> /dev/null)

DATASETS_URL := 'https://uclouvain-my.sharepoint.com/personal/victor_joos_uclouvain_be/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fvictor%5Fjoos%5Fuclouvain%5Fbe%2FDocuments%2FLDATA2010%2Fceleba%2Ezip' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-GB,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Referer: https://uclouvain-my.sharepoint.com/personal/victor_joos_uclouvain_be/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fvictor%5Fjoos%5Fuclouvain%5Fbe%2FDocuments%2FLDATA2010%2Fceleba%2Ezip&parent=%2Fpersonal%2Fvictor%5Fjoos%5Fuclouvain%5Fbe%2FDocuments%2FLDATA2010&ga=1' -H 'DNT: 1' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: iframe' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Connection: keep-alive' -H 'Cookie: MSFPC=GUID=cfd3872b330c49ebb54b9ea40dbd2ca3&HASH=cfd3&LV=202305&V=4&LU=1683631767027; MicrosoftApplicationsTelemetryDeviceId=94567cca-a5fe-4841-9ac1-415bb2386013; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEzLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uI2UzNjAxNmUzNDRhNTAwNDJjZWJmYTJjN2IxY2NhNDg0ODAxMGU3MTAyMjJhMTUwMTcyMzhiODY3NzY0YWExZDUsMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jZTM2MDE2ZTM0NGE1MDA0MmNlYmZhMmM3YjFjY2E0ODQ4MDEwZTcxMDIyMmExNTAxNzIzOGI4Njc3NjRhYTFkNSwxMzM0NTg4OTc1NTAwMDAwMDAsMCwxMzM0NTk3NTg1NTQ5MDkyOTcsMC4wLjAuMCwyNTgsNWE3MmQ3YjAtYTdiOS00YjU0LTkwZjgtNTgxYzk0Zjg1MTI0LCwsNjc2N2YzYTAtNzBjYi03MDAwLWI2MzctOTAyNGRmOTYwOGEwLDY3NjdmM2EwLTcwY2ItNzAwMC1iNjM3LTkwMjRkZjk2MDhhMCxBTnR4VWNFaHRVS1hRRFNIQ3FxYlpBLDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLCwxODkwMzIsR0FkeFdYM3FnLXBsUDRlOVhCUDF5MTZpZmpVLFV2MXQ3akVmb0NIUHVTTTVkaWkzenJKU3E5RTMrTEE4MmlkMThsTDNXYmxVMFFkWUkxcXBNeUdhSmtsUGRNNExZanROL0hpU3RwZHpjKzBXOTJWUXpIWVhDc1FzV0pDa0w4VW4wNnhqa2J3bFhRUUQxYlc1TDF5Z2JXbjA2Sm9LaEJ5TEZXaXF5eXhCYlFQNyttREgwMXR4R2lRZHVLaGRpc0FtZzlMZ2JKRUJCdllYZnRWVnppNE5tUGJDWW0xRmdkTG1kaU9NSTd1THdrYmZueWpVTGNuZXJtdUNhUTZaYU10bDlQYnZvOEJxU2RqS1NZMEhycHMySWpzUll1aXFjWWUzY3o4Tk14eG9mUy92WmdzdmNhWEZWaU9RdkN2OTN3YVNTTTdHczl0OWJMNEV4cWZxNmhYS2VuUXVuUjc4MWhXbE44YjA4U3Q0WXVqellDRWl2dz09PC9TUD4=; ai_session=tNt7nQlN/vs3alC7qCaYCw|1701415869303|1701415869307'
DATA_FOLDER := data
ASSETS_FOLDER := assets 

install: 
	@if [ -z $(POETRY) ]; then echo "Poetry could not be found. See https://python-poetry.org/docs/"; exit 2; fi
	$(POETRY) install
	$(POETRY) shell
	$(PIP) install "dash[diskcache]"
	$(PIP) install "dash[celery]"

venv:
	$(POETRY) shell

datasets:
	@if [ -z $(UNZIP) ]; then echo "Unzip could not be found. Please install it"; exit 2; fi
	curl $(DATASETS_URL) --output celeba.zip
	mkdir $(DATA_FOLDER)
	mkdir $(ASSETS_FOLDER)
	$(UNZIP) celeba.zip 
	mv celeba_buffalo_l.csv data/celeba_buffalo_l.csv 
	mv celeba_buffalo_s.csv data/celeba_buffalo_s.csv
	mv img_celeba assets/img_celeba
	rm -rf celeba.zip
	python precompute_dataset.py

precompute:
	python precompute_ndarray.py

run:
	python main.py